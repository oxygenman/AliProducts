import torch
import torch.nn as nn

class BaseLoss(nn.Module):
    """ Base loss
    """
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, x, gt):
        raise NotImplementedError

    def acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        acc_sum = torch.sum(preds == label)
        num = preds.shape[0]
        acc = acc_sum.float() / num * 100

        return acc
