import numpy as np
import argparse
import json
from PIL import Image
from os.path import join

Image.MAX_IMAGE_PIXELS = 2300000000
def fast_hist(a, b, n):#a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 1) & (a < n)#k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景）
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)#np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):#分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))#矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)


if __name__ == "__main__":
    name_classes = ['Background', 'Smoke', 'Corn', 'BarleyRice', 'Building']
    num_classes = 5
    hist = np.zeros((num_classes, num_classes))#hist初始化为全零，在这里的hist的形状是[19, 19]
    pred = np.array(Image.open(r"/root/hcj/workspace/test1/hcj_undergraduate_code/util/image_11_predict.png"))#读取一张图像分割结果，转化成numpy数组
    label = np.array(Image.open(r"/root/hcj/workspace/cutImage/SourceImageData/image_11_label.png"))#读取一张对应的标签，转化成numpy数组
    hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#对一张图片计算19×19的hist矩阵，并累加
    # x_ = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]])
    # x_[0, : ] = hist[1, : ]
    # print(x_)
    # hist = np.delete(hist, 0, axis=0)
    # print(hist)
    mIoUs = per_class_iu(hist)#计算所有验证集图片的逐类别mIoU值
    # print(mIoUs)
    for ind_class in range(1, 5):#逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs[1: 5]) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
