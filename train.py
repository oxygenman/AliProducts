# encoding = utf-8
"""
    Author: xuhaoyu@tju.edu.cn
    Github: https://github.com/misads
    License: MIT
"""
import os
import pdb
import time
import numpy as np
from collections.abc import Iterable

import torch
from torch import optim
from torch.autograd import Variable

import dataloader as dl
from options import opt
from scheduler import schedulers

from network import get_model
from eval import evaluate

from utils import *

from mscv.summary import create_summary_writer, write_meters_loss, write_image


import misc_utils as utils
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
# 初始化
with torch.no_grad():
    # 初始化路径
    save_root = os.path.join(opt.checkpoint_dir, opt.tag)
    log_root = os.path.join(opt.log_dir, opt.tag)

    utils.try_make_dir(save_root)
    utils.try_make_dir(log_root)

    train_dataloader = dl.train_dataloader
    val_dataloader = dl.val_dataloader

    # 初始化日志
    logger = init_log(training=True)

    # 初始化模型
    Model = get_model(opt.model)
    model = Model(opt)

    # 暂时还不支持多GPU
    # if len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model = model.to(device=opt.device)

    # 加载预训练模型，恢复中断的训练
    if opt.load:
        #saved_model = torch.load(opt.load)
        #model_dict = model.state_dict()
        #state_dict = {k:v for k,v in saved_model.items() if k in model_dict.keys()}
        #model_dict.update(state_dict)
        #model.load_state_dict(model_dict)
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume else 1
    else:
        start_epoch = 1

    # 开始训练
    model.train()

    # 计算开始和总共的step
    print('Start training...')
    start_step = (start_epoch - 1) * len(train_dataloader)
    global_step = start_step
    total_steps = opt.epochs * len(train_dataloader)
    start = time.time()

    #  定义scheduler
    optimizer = model.optimizer
    scheduler = model.scheduler

    # Tensorboard初始化
    writer = create_summary_writer(log_root)

    start_time = time.time()
    
    # 在日志记录transforms
    logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
    logger.info('===========================================')
    if val_dataloader is not None:
        logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
    logger.info('===========================================')

    # 在日志记录scheduler
    if opt.scheduler in schedulers:
        logger.info('scheduler: (Lambda scheduler)\n' + str(schedulers[opt.scheduler]))
        logger.info('===========================================')

try:
    # 训练循环
    eval_result = ''

    for epoch in range(start_epoch, opt.epochs + 1):
        for iteration, data in enumerate(train_dataloader):
            global_step += 1
            
            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            img, label = data['input'], data['label']  # ['label'], data['image']  #

            img_var = Variable(img, requires_grad=False).to(device=opt.device)
            label_var = Variable(label, requires_grad=False).to(device=opt.device)

            # 更新模型参数
            update = model.update(img_var, label_var)
            predicted = update.get('predicted')

            pre_msg = 'Epoch:%d' % epoch

            # 显示进度条
            msg = f'lr:{round(scheduler.get_lr()[0], 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len(train_dataloader), pre_msg, msg)
            # print(pre_msg, msg)

            # 训练时每1000个step记录一下summary
            if global_step % 1000 == 999:
                write_meters_loss(writer, 'train', model.avg_meters, global_step)

        # 每个epoch结束后的显示信息
        logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)

        # 训练中验证
        if epoch % opt.eval_freq == 0:

            model.eval()
            eval_result = evaluate(model, val_dataloader, epoch, writer, logger)
            model.train()

        if scheduler is not None:
            scheduler.step()


except Exception as e:

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的错误信息