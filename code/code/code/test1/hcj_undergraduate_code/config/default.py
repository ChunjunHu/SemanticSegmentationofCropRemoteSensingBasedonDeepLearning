'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
@LastEditTime: 2019-11-02 16:47:17
'''


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# #0. Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.CONFIG_FILE = ""
_C.LOG_PERIOD = float(0.03)      # print 
_C.VAL_PERIOD = float(1)
_C.TAG = "deeplabv3+"
# _C.TAG = "deeplabv3+modify"


# -----------------------------------------------------------------------------
# #1. DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()

## 1.1 DATA.DATASET
_C.DATA.DATASET = CN()
DATASET = _C.DATA.DATASET
DATASET.train_csv_file = r'/root/hcj/workspace/cutImage/TrainDataFinal/image_21.csv' 
DATASET.train_root_dir = r'/root/hcj/workspace/cutImage/TrainDataFinal/image/'
DATASET.train_mask_dir = r"/root/hcj/workspace/cutImage/TrainDataFinal/label/"
DATASET.val_csv_file =  r'/root/hcj/workspace/cutImage/ValData/image_20.csv'
DATASET.val_root_dir = r'/root/hcj/workspace/cutImage/ValData/image/'
DATASET.val_mask_dir = r"/root/hcj/workspace/cutImage/ValData/label/"

# DATASET.train_csv_file = r'/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/train.csv' 
# DATASET.train_root_dir = r'/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/image/'
# DATASET.train_mask_dir = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/label/"
# DATASET.val_csv_file =  r'/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/train.csv'
# DATASET.val_root_dir = r'/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/image/'
# DATASET.val_mask_dir = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/label/"

## 1.2 DATA.TRANSFORMS
_C.DATA.TRANSFORMS = CN()
TRANSFORMS = _C.DATA.TRANSFORMS
TRANSFORMS.ENABLE_NORMALIZE = True                       # std and mean
TRANSFORMS.Normalize = [(0.485,0.456,0.406),(0.229,0.224,0.225)]  # apply to both train and val
TRANSFORMS.ENABLE_RANDOMCHOICE = True
TRANSFORMS.RandomChoice = 1
TRANSFORMS.ENABLE_RANDOM_CROP = True
TRANSFORMS.RandomCrop = [1,(512,512)]

### 1.2.1 DATA.TRANSFORMS.TRAIN
TRANSFORMS.TRAIN = CN()
TRAIN = TRANSFORMS.TRAIN
TRAIN.GaussianBlur = [0.1,2]
TRAIN.RandomHorizontalFlip = [0.5] # p=0.2
TRAIN.RandomVerticalFlip = [0.5] # p=0.2
TRAIN.RandomChoce = [0.01,0.01,0.01,0.0]
# TRAIN.ColorJitter = [0.01,0.01,0.01,0.0]
# TRAIN.RandomErasing = [0.03,0.02,0.04,0.2]
TRAIN.Shift_Padding = [0.1,0.1,0.1] 


### 1.2.2 DATA.TRANSFORMS.VAL
TRANSFORMS.VAL = CN()
VAL = TRANSFORMS.VAL


## 1.3 DATA.DATALOADER
_C.DATA.DATALOADER = CN()
DATALOADER = _C.DATA.DATALOADER
# moren 42
DATALOADER.TRAIN_BATCH_SIZE = 21
DATALOADER.VAL_BATCH_SIZE = 21
# 计算使用的线程数默认12
DATALOADER.NUM_WORKERS = 30
DATALOADER.TRAIN_SHUFFLE = True
DATALOADER.VAL_SHUFFLE = False

# -----------------------------------------------------------------------------
# #2. ENGINE
# -----------------------------------------------------------------------------
_C.ENGINE = CN()


# -----------------------------------------------------------------------------
# #3. MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
MODEL = _C.MODEL
MODEL.NET_NAME = "deeplabv3plus_resnet101"
MODEL.LOAD_PATH = r"/root/hcj/workspace/test1/models/resnet101-5d3b4d8f.pth"         #file/path of the pretrain model
MODEL.DEVICE = 'cuda:0' 

# -----------------------------------------------------------------------------
# #4. OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
OUTPUT = _C.OUTPUT
OUTPUT.N_SAVED = 5
OUTPUT.DIR_NAME = r'./output/'

OUTPUT.ENABLE_RECODER = True
OUTPUT.VAL_METRICS = True
OUTPUT.TRAIN_METRICS = False
OUTPUT.TRAIN_LOSS = False
OUTPUT.VAL_LOSS = False

# -----------------------------------------------------------------------------
# #5. SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
SOLVER = _C.SOLVER
SOLVER.MAX_EPOCHS = 40

## 5.1 LOSS
SOLVER.CRITERION = "LabelSmoothing"
# SOLVER.CRITERION = "cross_entropy2d_drop_edge"

## 5.2 LR_SCHEDULER
SOLVER.ENABLE_LR_SCHEDULER = True
SOLVER.LR_SCHEDULER = "CosineAnnealingLR"
# SOLVER.LR_SCHEDULER = "ReduceLROnPlateau"
# SOLVER.LR_SCHEDULER = "StepLR"
SOLVER.LR_SCHEDULER_PATIENCE = 5
SOLVER.LR_SCHEDULER_FACTOR = 1/3
SOLVER.LR_SCHEDULER_REPEAT = 6
SOLVER.T_max = 12

## 5.3 OPTIMIZER
SOLVER.OPTIMIZER_NAME = "Adam"
SOLVER.LEARNING_RATE = 1e-4
SOLVER.SGD_MOMENTUM = 0.9
SOLVER.Adam_weight_decay = 0





# -----------------------------------------------------------------------------
# #6. tests
# -----------------------------------------------------------------------------
_C.TESTS = CN()
TESTS =  _C.TESTS





# -----------------------------------------------------------------------------
# #7. tools
# -----------------------------------------------------------------------------
_C.TOOLS = CN()
TOOLS = _C.TOOLS
TOOLS.image_n = 3
# TOOLS.zeros_1 = (50141,47161)
# TOOLS.zeros_2 = (46050,77470)
# TOOLS.zeros_3 = (19903,37241)
# TOOLS.zeros_4 = (28832,25936)
# TOOLS.root_dir_1_2 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/tests/stride_768/image_1_2/"
# TOOLS.root_dir_3_4 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/tests/stride_768/image_3_4/"
# TOOLS.csv_file_1 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/tests/stride_768/image_1.csv"
# TOOLS.csv_file_2 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/tests/stride_768/image_2.csv"
# TOOLS.csv_file_3 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/tests/stride_768/image_3.csv"
# TOOLS.csv_file_4 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/tests/stride_768/image_4.csv"
# TOOLS.image_4 = 4
# 电脑像素反过来，读取方式更人不一样
TOOLS.zeros_3 = (68363,18576)
# TOOLS.zeros_3 = (20767,42614)
TOOLS.zeros_4 = (29003,35055)
TOOLS.zeros_5 = (20115,43073)
TOOLS.zeros_6 = (21247,62806)
TOOLS.root_dir_3_4 = r"/root/hcj/workspace/cutImage/TestData/image/"
TOOLS.csv_file_3 = r"/root/hcj/workspace/cutImage/TestData/image_11.csv"
TOOLS.csv_file_4 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/round2/w1024s1024/val_image_4.csv"

# TOOLS.root_dir_3_4 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/round2/w1024s1024/image/"
# TOOLS.csv_file_3 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/round2/w1024s1024/val_image_3.csv"
# TOOLS.csv_file_4 = r"/home/LinHonghui/Datasets/tianchi_jinwei_AI2019/trainset/round2/w1024s1024/val_image_4.csv"

# TOOLS.root_dir_5_6 = r"./data/image"
# TOOLS.csv_file_5 = r"./data/image_5.csv"
# TOOLS.csv_file_6 = r"./data/image_6.csv"

TOOLS.shuffle = False
# 24
TOOLS.batch_size = 24
# 默认8
TOOLS.num_workers = 30

TOOLS.save_path = r"/root/hcj/workspace/test1/hcj_undergraduate_code/output/savedResult/"

# -----------------------------------------------------------------------------
# #8. utils
# -----------------------------------------------------------------------------
_C.UTILS = CN()
UTILS = _C.UTILS
UTILS.METRICS = "Label_Accuracy_delete_edge"





if __name__ == "__main__":
    output_path = "./configs/default.yml"
    cfg = _C.dump()
    print(cfg)
    out = open(output_path,'w')
    out.write(cfg)
    out.close()
