import torch
import torch.nn as nn
from torch import Tensor

class InputEquivariantLayer(nn.Module):

  def __init__(self, in_features, out_features, num_inputs, func):
    super(InputEquivariantLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.num_inputs = num_inputs

    funcs = {'tanh':nn.Tanh(),
             'softplus':nn.Softplus(),
             'sigmoid':nn.Sigmoid(),
             'elu':nn.ELU(),
             'celu':nn.CELU()}

    self.fc = nn.Linear(self.in_features, self.out_features, bias=True)
    self.af = funcs[func.lower()]

  def forward(self, h):
    g = h.mean(dim=1, keepdim=True).repeat(1,self.num_inputs)
    f = torch.stack((h,g), dim=2)
    return self.af(self.fc(f))

class IntermediateEquivariantLayer(nn.Module):

  def __init__(self, in_features, out_features, num_inputs, func):
    super(IntermediateEquivariantLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.num_inputs = num_inputs

    funcs = {'tanh':nn.Tanh(),
             'softplus':nn.Softplus(),
             'sigmoid':nn.Sigmoid(),
             'elu':nn.ELU(),
             'celu':nn.CELU()}

    self.fc = nn.Linear(self.in_features, self.out_features, bias=True)
    self.af = funcs[func.lower()]

  def forward(self, h):
    g = h.mean(dim=1, keepdim=True).repeat(1,self.num_inputs,1)
    f = torch.cat((h,g), dim=2)
    return self.af(self.fc(f))
    
class SLogSlaterDeterminant(nn.Module):

  def __init__(self, in_features: int, num_inputs: int, bias: bool):
    super(SLogSlaterDeterminant, self).__init__()
    self.in_features = in_features
    self.num_inputs = num_inputs

    self.weight = nn.Parameter(torch.Tensor(self.in_features, self.num_inputs))
    if(bias==True):
      self.bias = nn.Parameter(torch.Tensor(self.num_inputs))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()
    self.log_factorial = torch.arange(1,self.num_inputs+1, dtype=torch.float32).log().sum() 

  def reset_parameters(self):
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.nn.init.zeros_(self.bias)

  def forward(self, input):
    slater_matrix = torch.matmul(input, self.weight) + self.bias
    sign, logabsdet = torch.slogdet(slater_matrix)
    return sign, logabsdet - 0.5*self.log_factorial

class myFunction(nn.Module):
  
  def __init__(self, num_inputs: int, num_hidden: int, num_layers: int , func: str):
    super(myFunction, self).__init__()
    """
    A R^N -> R^1 function, which returns the log abs. determinant (along with its sign)
    of a real valued input.
    """
    self.num_inputs = num_inputs        #number of input nodes
    self.num_hidden = num_hidden        #number of hidden units per layer
    self.num_layers = num_layers        #number of layers for network
    self.func = func

    layers = [] #list to store all the layers
    
    #first layer is an input equivariant layer
    self.input_layer = InputEquivariantLayer(in_features=2,
                                    out_features=self.num_hidden,
                                    num_inputs=self.num_inputs,
                                    func=self.func)

    #remainding layers are intermediate version (only differ by torch.stack/torch.cat op)
    for i in range(1,self.num_layers):
      layers.append(IntermediateEquivariantLayer(in_features=2*self.num_hidden,
                                   out_features=self.num_hidden,
                                   num_inputs=self.num_inputs,
                                   func=self.func))

    self.layers = nn.ModuleList(layers) #convert to ModuleList

    #Layer which converts the output features to a single value
    #via use of Signed-Log determinant
    self.slater = SLogSlaterDeterminant(in_features=self.num_hidden,
                                  num_inputs=self.num_inputs,
                                  bias=True)
    #the width of a Gaussian envelope function - this restricts the output of the 
    #signed-log determinant forcing it to zero at large input values
    self.width = nn.Parameter(torch.empty(1).fill_(0.1))
    
  def forward(self, x0):
    x = self.input_layer(x0)
    for l in self.layers:
      x = l(x) + x
    log_bcs = -self.width*x0.pow(2).sum(dim=1) #gaussian envelope (to restrict output)
    sign, logabsdet = self.slater(x)
    return sign, logabsdet + log_bcs
    
def laplacian_from_log_domain(xs: Tensor, network: nn.Module):
  """ 
  Computes the laplacian of a given function (within the log-domain as opposed to
  the linear domain for numerical stability). 
  This function computes (1/f)*(d2f/dx2) which is equivalent to d2log(f)/dx2 + (dlog(f)/dx)^2
  """
  xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
  xs_flat = torch.stack(xis, dim=1)
  sign, ys = network(xs_flat.view_as(xs))

  ones = torch.ones_like(ys)
  (dy_dxs,) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)

  lap_ys = sum(
      torch.autograd.grad(
          dy_dxi, xi, ones, retain_graph=True, create_graph=False)[0]
      for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))
  )

  return lap_ys + dy_dxs.pow(2).sum(dim=-1)
    
num_inputs = 4
num_hidden = 64
num_layers = 2
func = 'tanh'
    
myFunc = myFunction(num_inputs=num_inputs,
                    num_hidden=num_hidden,
                    num_layers=num_layers,
                    func=func)
                    
batch = 100000
                    
X = torch.randn(batch, num_inputs)

sign, logabsdet = myFunc(X)

    
d2y_dx2 = laplacian_from_log_domain(X, myFunc)


print("X: ",X.shape)
print("sign/logabsdet: ",sign.shape, logabsdet.shape)
print("laplacian: ",d2y_dx2.shape)
