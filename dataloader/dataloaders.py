# encoding=utf-8
from dataloader.products import *
from dataloader.transforms import get_transform
from torch.utils.data import DataLoader
from options import opt
import pdb
import torch
import torch.utils.data.sampler as Sampler
###################

TEST_DATASET_HAS_OPEN = True  # 有没有开放测试集

###################

train_list = "./datasets/final_train.txt"
val_list = "./datasets/val.txt"

max_size = 128 if opt.debug else None  # debug模式时dataset的最大大小

# transforms
transform = get_transform(opt.transform)
train_transform = transform.train_transform
val_transform = transform.val_transform

# datasets和dataloaders
train_dataset = TrainValDataset(train_list, transforms=train_transform, max_size=max_size)
####添加平衡采样
# np.bincount计算相对应的label的个数,再转换成list
classcount = np.bincount(train_dataset.labels).tolist()
print('classcount size:',len(classcount))
print('classcount:',classcount)
# 设置权重
train_weights = 1./torch.tensor(classcount, dtype=torch.float)
train__sampleweights = train_weights[train_dataset.labels]
# 注意,这里的num_samples就是等于train__sampleweights的长度

#print("注意：使用weighted random sampler")
train_sampler = Sampler.WeightedRandomSampler(weights=train__sampleweights, num_samples=len(train__sampleweights))
#trainloader = DataLoader(trainset, sampler=train_sampler, shuffle=False)
###
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,sampler=train_sampler,num_workers=opt.workers, drop_last=True)
#train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size,shuffle=False,num_workers=opt.workers, drop_last=False)
val_dataset = TrainValDataset(val_list, transforms=val_transform, max_size=max_size)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers//2)

if TEST_DATASET_HAS_OPEN:
    test_list = "./datasets/test.txt"  # 测试集

    test_dataset = TestDataset(test_list, transforms=val_transform, max_size=max_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

else:
    test_dataloader = None
