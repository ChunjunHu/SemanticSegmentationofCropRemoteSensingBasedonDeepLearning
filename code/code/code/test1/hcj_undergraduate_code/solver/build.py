'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-08-26 17:14:27
'''
import solver.criterion as Criterion
from importer import *


def make_criterion(cfg):
    SOLVER = cfg.SOLVER
    if hasattr(torch.nn,SOLVER.CRITERION):
        criterion = getattr(torch.nn,SOLVER.CRITERION)()
    elif hasattr(Criterion,SOLVER.CRITERION):
        criterion = getattr(Criterion,SOLVER.CRITERION)()
    else:
        raise TypeError("Loss function not found. Got {}".format(SOLVER.CRITERION))
    return criterion

def make_optimizer(cfg,model):
    SOLVER = cfg.SOLVER
    lr = SOLVER.LEARNING_RATE
    # 获取动量防止震荡
    momentum = SOLVER.SGD_MOMENTUM
    # 获取模型参数
    params = model.parameters()
    if SOLVER.OPTIMIZER_NAME == "Adam":
        # params待优化参数的iterable或者是定义了参数组的dict,betas用于计算梯度以及梯度平方的运行平均值的系数,eps防止除以0很小的数,weight_decay衰减权重
        optimizer = optim.Adam(params,lr=lr,weight_decay=SOLVER.Adam_weight_decay)
    elif SOLVER.OPTIMIZER_NAME == "SGD":
        optimizer = getattr(optim,SOLVER.OPTIMIZER_NAME)(params,lr=lr,momentum=momentum)
    else:
        optimizer = getattr(optim,SOLVER.OPTIMIZER_NAME)(params,lr=lr)

    return optimizer

def make_lr_scheduler(cfg,optimizer):
        SOLVER = cfg.SOLVER
        factor = SOLVER.LR_SCHEDULER_FACTOR
        patience = SOLVER.LR_SCHEDULER_PATIENCE
        if SOLVER.ENABLE_LR_SCHEDULER:
            if SOLVER.LR_SCHEDULER == "StepLR":
                lr_scheduler = StepLR(optimizer,patience,factor)
            elif SOLVER.LR_SCHEDULER == "ReduceLROnPlateau":
                lr_scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=patience,factor=factor)
            elif SOLVER.LR_SCHEDULER == "CosineAnnealingLR":
                lr_scheduler = CosineAnnealingLR(optimizer,T_max=SOLVER.T_max,eta_min=1e-8)
        return lr_scheduler