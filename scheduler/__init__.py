from torch import optim
from easydict import EasyDict
from options import opt


warmup = 1  # warmup的学习率系数, 为1时不进行warmup

schedulers = {
    '1x': {  # Faster_RCNN
        'epochs': [1, 10, 11, 14],  # 12个epoch   7:3:2
        'ratios': [warmup, 1, 0.1, 0.01],
    },
    '2x': {
        'epochs': [2, 14, 20, 24],  # 24个epoch
        'ratios': [warmup, 1, 0.1, 0.01],
    },
    '5x': { 
        'epochs': [3, 35, 50, 60],  # 60个epoch
        'ratios': [warmup, 1, 0.1, 0.01],
    },
    '10x': {  
        'epochs': [5, 70, 100, 120],  # 120个epoch
        'ratios': [warmup, 1, 0.1, 0.01],
    },
    '20e': {
        'epochs': [1, 15, 18, 20],  # 20个epoch
        'ratios': [warmup, 1, 0.1, 0.01],
    },
    '100e': {  # Yolo2和Yolo3
        'epochs': [5, 60, 90, 100],  # 100个epoch
        'ratios': [warmup, 1, 0.1, 0.01],
    }

}


schedulers = EasyDict(schedulers)

def get_scheduler(opt, optimizer):
    if opt.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr * 0.1)
    elif opt.scheduler.lower() == 'none':
        def lambda_decay(step) -> float:
            return 1.
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)
    else:
        if opt.scheduler not in schedulers:
            if opt.scheduler[-1] == 'x': 
                times = int(opt.scheduler[:-1])
                schedulers[opt.scheduler] = {
                    'epochs': [s * times for s in schedulers['1x'].epochs],
                    'ratios': [warmup, 1, 0.1, 0.01],
                }

        epochs = schedulers[opt.scheduler].epochs  # [1, 7, 10, 99999999]
        opt.epochs = epochs[-1]
        ratios = schedulers[opt.scheduler].ratios
        
        def lambda_decay(step) -> float:
            for epoch, ratio in zip(epochs, ratios):
                if step < epoch:
                    return ratio
            return ratios[-1]

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_decay)        

    return scheduler
