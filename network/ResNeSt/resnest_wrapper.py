import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalized_linear import NormalizedLinear
from .resnest import resnest50, resnest101, resnest200
from options import opt

classes = opt.num_classes

arch_dict = {
    'ResNeSt50': resnest50,
    'ResNeSt101': resnest101,
    'ResNeSt200': resnest200

}

feature_len=512
class Classifier(nn.Module):
    def __init__(self, arch):
        super(Classifier, self).__init__()
        self.network = arch_dict[arch](pretrained=True)
        # pdb.set_trace()
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, feature_len)
        self.normalized_fc=NormalizedLinear(feature_len,classes)
        #self.bn= nn.BatchNorm1d(feature_len)
        #self.bn.bias.requires_grad_(False)

    def forward(self, input):
        x = input
        feature= self.network(x)
        #feature = self.bn(feature)
        #print("bn output:",feature)
        y=self.normalized_fc(feature)
        #print("normalized output:",y[0])
        return feature,y

#a = Classifier('ResNeSt101')
#img = torch.randn([2, 3, 256, 256])
#a(img)
