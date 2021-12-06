import pdb
import sys
import numpy as np
import torch
import torch.nn as nn
import os
from torch import optim
from mscv import ExponentialMovingAverage, print_network
from network.base_model import BaseModel

from loss import get_arc_loss
from optimizer import get_optimizer
from scheduler import get_scheduler

from .resnest_wrapper import Classifier

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        if opt.gpu_num>1:
            self.classifier = nn.DataParallel(Classifier(opt.model))
            print("-------------------use gpu-------------------")
        else:
            self.classifier = Classifier(opt.model)  
        # self.classifier.apply(weights_init)  # 初始化权重

        print_network(self.classifier)

        self.optimizer = get_optimizer(opt, self.classifier)
        self.scheduler = get_scheduler(opt, self.optimizer)


    def update(self, input, label):
        feature,predicted = self.classifier(input)

        loss,acc= get_arc_loss(predicted, label, avg_meters=self.avg_meters)
        #if np.isnan(loss.cpu().detach().numpy()):
        #    sys.exit(0)
        #print（"----------",loss）

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'predicted': predicted}

    def forward(self, x):
        return self.classifier(x)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)