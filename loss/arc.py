import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLoss
from .normalized_linear import NormalizedLinear
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)     # 必须要torch.LongTensor()
    return true_dist
class ArcLoss(BaseLoss):
    """ Arc loss
    """
    def __init__(self, 
                 scale=100.0,
                 margin=0.5):
        super(ArcLoss, self).__init__()

        self.scale = scale
        self.margin = margin
        #self.input_dim = input_dim
        #self.output_dim = output_dim

        #self.cls = NormalizedLinear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, cos_theta, gt):
        #cos_theta = self.cls(x)
        #print(cos_theta)
        one_hot = torch.zeros(cos_theta.size()).cuda()
        one_hot.scatter_(1, gt.view(-1, 1).long().cuda(), 1)

        theta = torch.acos(cos_theta)
        theta_plus_margin = theta + self.margin
        cos_theta_plus_margin = torch.cos(theta_plus_margin)
    
        logit = one_hot * cos_theta_plus_margin + (1 - one_hot) * cos_theta
        logit.mul_(self.scale) 
        acc = self.acc(cos_theta, gt.cuda())
        loss = self.criterion(logit+1e-9, gt.cuda())
        #print('loss:',loss)
        return loss,acc
