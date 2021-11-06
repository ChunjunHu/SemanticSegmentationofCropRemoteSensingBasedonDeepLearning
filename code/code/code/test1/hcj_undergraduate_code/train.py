'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
@LastEditTime: 2019-09-02 16:44:20
'''
from argparse import ArgumentParser
from os import mkdir,path
import sys
sys.path.append('..')
# import os
# print(os.getcwd())
from config import cfg
from config import *
from data.dataloader import make_dataloader,make_inference_dataloader
from engine import do_train
from model import build_model
from solver import make_optimizer
from solver import make_criterion
from solver import make_lr_scheduler
from util import make_metrics
from importer import *


def train(cfg):
    #建立学习网络使用deeplabv3resnet101
    model = build_model(cfg)
    # 通过nn.dataParallel并行加速计算
    model = nn.DataParallel(model)
    # 开启GPU并行加速计算
    torch.backends.cudnn.benchmark = True
    # 确定学习率的优化器
    optimizer = make_optimizer(cfg,model)
    # 确定损失函数
    criterion = make_criterion(cfg)
    # 调整学习率
    scheduler = make_lr_scheduler(cfg,optimizer)
    # 设置模型评估标准
    metrics = make_metrics(cfg)
    # 载入数据
    train_loader = make_dataloader(cfg,is_train=True)
    val_loader = make_dataloader(cfg,is_train=False)
    
    cfg.TOOLS.image_n = 3
    cfg.TOOLS.image_n = 4

    do_train(cfg,model=model,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,
                    scheduler=scheduler,loss_fn=criterion,metrics=metrics)


#对模型的参数进行初始化
def load_arg():
    parser = ArgumentParser(description="Pytorch Hand Detection Training")
    parser.add_argument("-config_file","--CONFIG_FILE",type=str,help="Path to config file")
    parser.add_argument('-log_period',"--LOG_PERIOD",type=float,help="Period to log info")
    parser.add_argument("-val_period","--VAL_PERIOD",type=float)
    parser.add_argument("-tag","--TAG",type=str)
    
    # DATA
    # Batch大小是在更新模型之前处理的多个样本, Epoch数是通过训练数据集的完整传递次数
    parser.add_argument("-num_workers", "--DATA.DATALOADER.NUM_WORKERS",type=int,
                        help='Num of data loading threads. ')
    parser.add_argument("-train_batch_size","--DATA.DATALOADER.TRAIN_BATCH_SIZE",type=int,
                        help="input batch size for training (default:64)")
    parser.add_argument("-val_batch_size","--DATA.DATALOADER.VAL_BATCH_SIZE",type=int,
                        help="input batch size for validation (default:128)")
    parser.add_argument("-train_csv_file","--DATA.DATASET.train_csv_file",type=str)
    parser.add_argument("-train_root_dir","--DATA.DATASET.train_root_dir",type=str)
    parser.add_argument("-train_mask_dir","--DATA.DATASET.train_mask_dir",type=str)
    parser.add_argument("-val_csv_file","--DATA.DATASET.val_csv_file",type=str)
    parser.add_argument("-val_root_dir","--DATA.DATASET.val_root_dir",type=str)
    parser.add_argument("-val_mask_dir","--DATA.DATASET.val_mask_dir",type=str)

    # MODEL
    parser.add_argument('-model',"--MODEL.NET_NAME",type=str,
                        help="Net to build")
    parser.add_argument('-path',"--MODEL.LOAD_PATH",type=str,
                        help="path/file of a pretrain model(state_dict)")
    parser.add_argument("-device","--MODEL.DEVICE",type=str,
                        help="cuda:x (default:cuda:0)")

    # SOLVER
    parser.add_argument("-max_epochs","--SOLVER.MAX_EPOCHS",type=int,
                        help="num of epochs to train (default:50)")
    parser.add_argument('-optimizer',"--SOLVER.OPTIMIZER_NAME",type=str,
                        help="optimizer (default:SGD)")
    parser.add_argument("-criterion","--SOLVER.CRITERION",type=str,
                        help="Loss Function (default: GIoU_L1Loss)")
    parser.add_argument("-lr","--SOLVER.LEARNING_RATE",type=float,
                        help="Learning rate (default:0.01)")
    parser.add_argument('-patience','--SOLVER.LR_SCHEDULER_PATIENCE',type=int,
                        help='Number of events to wait if no improvement and then stop the training. (default:100)')
    parser.add_argument('-factor','--SOLVER.LR_SCHEDULER_FACTOR',type=float,
                        help='factor of lr_scheduler (default: 1/3)')
    parser.add_argument("-lr_scheduler","--SOLVER.LR_SCHEDULER",type=str)


    # OUTPUT 
    parser.add_argument("-n_saved","--OUTPUT.N_SAVED",type=int)

    # TOOLS
    parser.add_argument("-image_n","--TOOLS.image_n",type=int,default=3)
    parser.add_argument("-save_path","--TOOLS.save_path",type=str)
    # 解析添加的参数
    arg = parser.parse_args()
    return arg
# 对参数形成键值对
def merge_from_dict(cfg,arg_dict):
    for key in arg_dict:
        if arg_dict[key] != None:
            cfg.merge_from_list([key,arg_dict[key]])
    return cfg

def seed_torch(seed=15):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置初始化深度学习的随机种子，保证每次计算的公平性
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 为GPU设置随机种子
    torch.cuda.manual_seed(seed)
    # 调用的CuDNN的卷积操作保持一致
    torch.backends.cudnn.deterministic = True

if  __name__ == "__main__":
    seed_torch()
    arg = load_arg()
    # 从命令行使用覆盖选项
    if arg.CONFIG_FILE != None:
        cfg.merge_from_file(arg.CONFIG_FILE)

    cfg = merge_from_dict(cfg,vars(arg))
    print(cfg)
    # 如果输出的模型名存在但是路径不存在，则在根目录下建立一个名为模型名的文件夹
    if cfg.OUTPUT.DIR_NAME and not path.exists(cfg.OUTPUT.DIR_NAME):
        mkdir(cfg.OUTPUT.DIR_NAME)
    
    train(cfg)