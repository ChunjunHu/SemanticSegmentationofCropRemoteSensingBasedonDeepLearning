'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 去除连通域和空洞在推理之后进行的后处理步骤
LastEditTime: 2019-09-04 15:02:41
'''
import os
import threading
import cv2 as cv
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from argparse import ArgumentParser
# from keras.utils import to_categorical
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000000000
#将类向量转化为类矩阵
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
    
class MyThread(threading.Thread):

    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None
        
def label_resize_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签 
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    label = cv.resize(label.copy(),None,fx=0.1,fy=0.1)
    r = np.where(label == 1, 255, 0)
    g = np.where(label == 2, 255, 0)
    b = np.where(label == 3, 255, 0)
    yellow = np.where(label == 4, 255, 0)
    anno_vis = np.dstack((b, g, r)).astype(np.uint8)
    # 黄色分量(红255, 绿255, 蓝0)
    anno_vis[:, :, 0] = anno_vis[:, :, 0] + yellow
    anno_vis[:, :, 1] = anno_vis[:, :, 1] + yellow
    anno_vis[:, :, 2] = anno_vis[:, :, 2] + yellow
    if img is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(img, alpha, anno_vis, 1-alpha, 0)
        return overlapping

def remove_small_objects_and_holes(class_type,label,min_size,area_threshold,in_place=True):
    print("------------- class_n : {} start ------------".format(class_type))
    if class_type == 4:
        # kernel = cv.getStructuringElement(cv.MORPH_RECT,(500,500))
        # label = cv.dilate(label,kernel)
        #kernel = cv.getStructuringElement(cv.MORPH_RECT,(10,10))
        #label = cv.erode(label,kernel)
        label = remove_small_objects(label==1,min_size=min_size,connectivity=1,in_place=in_place)
        label = remove_small_holes(label==1,area_threshold=area_threshold,connectivity=1,in_place=in_place)
    else:
        label = remove_small_objects(label==1,min_size=min_size,connectivity=1,in_place=in_place)
        label = remove_small_holes(label==1,area_threshold=area_threshold,connectivity=1,in_place=in_place)        
    print("------------- class_n : {} finished ------------".format(class_type))
    return label

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-image_n",type=int,help="传入5或6，指定",default=11)
    parser.add_argument("-image_path",type=str,help="传入image_n_predict所在路径",default=r"/root/hcj/workspace/test1/hcj_undergraduate_code/output/savedResult/image_3_predict.png")
    parser.add_argument("-threshold",type=int,default=20000)
    arg = parser.parse_args()
    image_n = arg.image_n
    image_path = arg.image_path
    threshold = arg.threshold
    
    if image_n == 11:
        source_image = cv.imread(r"/root/hcj/workspace/cutImage/vis/image_11_vis.png")
        mask = np.asarray(Image.open(r"/root/hcj/workspace/cutImage/vis/image_11_mask.png"))
    else:
        raise ValueError("image_n should be 11, Got {} ".format(image_n))

    image = np.asarray(Image.open(image_path))
    
    label = to_categorical(image, num_classes=5,dtype='uint8')


    threading_list = []
    for i in range(5):
        t = MyThread(remove_small_objects_and_holes,args=(i,label[:,:,i],threshold,threshold,True))
        threading_list.append(t)
        t.start()

    
    # 等待所有线程运行完毕
    result = []
    for t in threading_list:
        t.join()
        result.append(t.get_result()[:,:,None])
    
    label = np.concatenate(result,axis=2)

    label = np.argmax(label, axis=2).astype(np.uint8)
    cv.imwrite('image_'+str(image_n)+"_predict.png", label)
    mask = label_resize_vis(label,source_image)
    cv.imwrite('vis_image_'+str(image_n)+"_predict.jpg", mask)
