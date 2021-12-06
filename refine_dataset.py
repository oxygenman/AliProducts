# python 3.5, pytorch 1.14

import os, sys
import ipdb
from collections import defaultdict

#import dataloader as dl
from options import opt
from mscv.summary import write_loss

import torch
import torchvision
import numpy as np
import subprocess
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from torch.utils.data import DataLoader
from collections.abc import Iterable
from PIL import Image
from utils import *
from loss.arc import  ArcLoss
import misc_utils as utils
from options import opt
from network import get_model
import misc_utils as utils
root_path='/home/young.xu/data/aliproducts/fast_features/train'
from sklearn.metrics.pairwise import cosine_similarity
def generate_features():
    if not opt.load:
        print('Usage: refine_dataset.py [--tag TAG] --load LOAD')
    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)
    load_epoch = model.load(opt.load)
    if load_epoch is not None:
        opt.which_epoch = load_epoch
    model.eval()
    #加载训练数据集 
    dataloader=dl.train_dataloader
    arc_loss=ArcLoss()
    for i, data in enumerate(dataloader):
        image, label, path = data['input'], data['label'], data['path']
        utils.progress_bar(i, len(dataloader), 'Eva... ')
        # ct_num += 1
        with torch.no_grad():
            txt_path=root_path+path[0][-18:]+'.txt'
            dir_path=os.path.dirname(txt_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(txt_path,'w') as single_f:
                img_var = Variable(image, requires_grad=False).to(device=opt.device)
                label_var = Variable(label, requires_grad=False).to(device=opt.device)
                feature,predicted = model(img_var)
                loss,acc = arc_loss(predicted,label_var)
                #print('loss:',loss)
                #print(feature)
                #print(feature.cpu().numpy().tolist()[0])
                single_f.write(str(np.squeeze(feature.cpu().numpy()).tolist()))
                single_f.write('\n')
                single_f.write(str(loss.cpu().numpy().tolist()))

def generate_features_fast():
    if not opt.load:
        print('Usage: refine_dataset.py [--tag TAG] --load LOAD')
    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)
    load_epoch = model.load(opt.load)
    if load_epoch is not None:
        opt.which_epoch = load_epoch
    model.eval()
    #加载训练数据集 
    dataloader=dl.train_dataloader
    arc_loss=ArcLoss()
    for i, data in enumerate(dataloader):
        image, label, path = data['input'], data['label'], data['path']
        utils.progress_bar(i, len(dataloader), 'Eva... ')
        # ct_num += 1
        with torch.no_grad():
            #batch inference 
            img_var = Variable(image, requires_grad=False).to(device=opt.device)
            label_var = Variable(label, requires_grad=False).to(device=opt.device)
            feature,predicted = model(img_var)
            #loss,acc = arc_loss(predicted,label_var)
            for idx in range(len(path)):
                 txt_path=root_path+path[idx][-18:]+'.txt'
                 dir_path=os.path.dirname(txt_path)
                 if not os.path.exists(dir_path):
                     os.makedirs(dir_path)
                 with open(txt_path,'w') as single_f:
                     single_f.write(str(np.squeeze(feature[idx].cpu().numpy()).tolist()))
                     single_f.write('\n')
                     #single_f.write(str(loss[idx].cpu().numpy().tolist()))
def filter_images_by_cos_similarity(root_path,threshold):
    hard_file_path='/home/young.xu/data/aliproducts/fast_features/hard_file.txt'
    result_path='/home/young.xu/data/aliproducts/fast_features/test_filter_train.txt'
    except_path='/home/young.xu/data/aliproducts/fast_features/err.txt'
    except_file=open(except_path,'w')
    result_file=open(result_path,'w')
    for one_dir in os.listdir(root_path):
        #print(one_dir)
        file_num = len(os.listdir(os.path.join(root_path,one_dir)))
        #print('before filter num:',file_num)
        feature_array = np.zeros((file_num,512), dtype=np.float32)
        file_list=[]
        for i,one_file in enumerate(os.listdir(os.path.join(root_path,one_dir))):
            file_path=os.path.join(root_path,one_dir,one_file)
            #print(file_path)
            file_list.append(file_path)
            try:
                feature_list=open(file_path).readlines()[0].strip('[]\n').split(',')
               # print(feature_list)
            except:
                except_file.write(one_dir+'\n')
                continue
            else:
                feature_array[i]=np.array(feature_list)
        cos_matrix=cosine_similarity(feature_array)
        avg_cos_array=cos_matrix.sum(axis=0)/file_num
        #print(avg_cos_array)
        result_index=np.where(avg_cos_array>threshold)[0]
        result_num = len(result_index)
        #print('after filter num:',result_num)
        if result_num==0:
            print(one_dir)
            result_index=range(file_num)
        #print(result_index)
        for index in result_index:
            result_file.write(file_list[index][:-4]+" "+one_dir+'\n')
    result_file.close()
    except_file.close()
if __name__ == '__main__':
    root_path='/home/young.xu/data/aliproducts/fast_features/train'
    filter_images_by_cos_similarity(root_path,0.2)
    #generate_features_fast()







