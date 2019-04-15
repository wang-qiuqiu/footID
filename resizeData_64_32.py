import os
from scipy import ndimage
import numpy as np
import scipy
import cv2


# 将原始图片都resize到64*32
def resize_save_data(source, target):
    pic_count = 0
    if not os.path.exists(target):
        os.makedirs(target)
    for folder in os.listdir(source):
        temp = os.path.join(source, folder)
        if not os.path.exists(os.path.join(target, folder)):
            os.makedirs(os.path.join(target, folder))
        for pic in os.listdir(temp):
            pic_dir = os.path.join(temp, pic)
            image = np.array(ndimage.imread(pic_dir, flatten=False))
            image = scipy.misc.imresize(image, size=(64, 32))
            target_ = os.path.join(target, folder)
            save_path = target_ + '\\' + pic
            if os.path.exists(save_path):
                os.remove(save_path)
            cv2.imwrite(save_path, image)
            pic_count += 1
            print(str(pic_count) + ' @' + '缩放鞋印图像：' + pic_dir + '完成')


# 一个文件夹下存放所有图片
def resize_save_data_from_img(source, target):
    pic_count = 0
    if os.path.exists(target):
        for file in os.listdir(target):
            path = os.path.join(target, file)
            os.remove(path)
        os.rmdir(target)
    os.makedirs(target)
    for pic in os.listdir(source):
        pic_path = os.path.join(source, pic)
        image = np.array(ndimage.imread(pic_path, flatten=False))
        image = scipy.misc.imresize(image, size=(64, 32))
        save_path = target + '\\' + pic
        cv2.imwrite(save_path, image)
        pic_count += 1
        print(str(pic_count) + ' @' + '缩放鞋印图像：' + pic + '完成')


if __name__ == '__main__':
    source_dir = 'C:\\Users\\Neo\\Desktop\\outlier'
    save_dir = 'C:\\Users\\Neo\\Desktop\\outlier_test'
    resize_save_data(source_dir, save_dir)