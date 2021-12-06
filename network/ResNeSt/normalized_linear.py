import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-5

class NormalizedLinear(nn.Module):
    """ Normalized linear module
    """
    def __init__(self,
                 input_dim, 
                 output_dim):
        super(NormalizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim).cuda())
        nn.init.normal_(self.weight, 0, 0.1)

    def forward(self, x):
       
        #print('-----------x:',x)
        #print('************weight:',self.weight)
        cos = F.linear(F.normalize(x), F.normalize(self.weight))
        cos=cos.clamp(-1+eps,1+eps)

        return cos