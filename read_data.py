import os
import xlwt
import shutil
from scipy import ndimage
import numpy as np
import scipy
import cv2

fileDir = 'C:\\Users\\Neo\\Desktop\\source'
target_dir_root = 'C:\\Users\\Neo\\Desktop\\train'
targets = []


####    需要安装两个包：xlwt和xlrd
#   统计文件夹下图片的数量
def statisticDataNum(fileDir, file):
    dataNum = 0
    temp_dir = os.path.join(fileDir, file)
    for pics in os.listdir(temp_dir):
        if pics.endswith('.jpg'):
            dataNum += 1
        else:
            os.remove(os.path.join(temp_dir, pics))
    print(str(file) + ": " + str(dataNum))
    return dataNum


#   创建xls
def createXls():
    workbook = xlwt.Workbook(encoding='utf-8')
    return workbook


#   写入xls
def write2Xls(dir, workbook):
    worksheet = workbook.add_sheet('鞋印数据统计', cell_overwrite_ok=True)
    worksheet.write(0, 0, '类别')
    worksheet.write(0, 1, '数量')
    row = 1
    col = 0
    i = 0
    for file in os.listdir(dir):
        dataNums = statisticDataNum(dir, file)
        if dataNums > 2 :#and i < 300:
            # 将数量为16的图片拷贝到指定目录中去，拷贝50类
            # if dataNums == 16 and i < 50:
            #     temp_dir = os.path.join(dir, file)
            copyData(os.path.join(dir, file), file)
            #     deleteFolder(temp_dir)
            #     i += 1
            worksheet.write(row, col, file)
            worksheet.write(row, col+1, dataNums)
            row += 1
            i += 1


# 将数据拷贝出去以后删除文件夹
def deleteFolder(folder_dir):
    for file in os.listdir(folder_dir):
        os.remove(os.path.join(folder_dir, file))
    os.rmdir(folder_dir)


#   将满足数量条件的文件夹拷贝到指定目录下
def copyData(source_dir_root, source_dir):
    i = 1
    print('data copy from: ' + str(source_dir_root))
    target_dir = os.path.join(target_dir_root, str(source_dir))
    if not os.path.exists(target_dir):
        os.makedirs(os.path.join(target_dir_root, str(source_dir)))
    for src_pic in os.listdir(source_dir_root):
        if not os.path.exists(os.path.join(target_dir, str(src_pic))):
            shutil.copy(os.path.join(source_dir_root, src_pic), target_dir)
            # resize_save_data(target_dir, src_pic)
            print('pic: ' + str(i) + ' @ ' + 'copy footprint: ' + str(src_pic))
        i += 1
    print('copy finished from: ' + str(source_dir_root))


# resize数据到32*64
def resize_save_data(source, name):
    source_dir = os.path.join(source, name)
    image = np.array(ndimage.imread(source_dir, flatten=False))
    image = scipy.misc.imresize(image, size=(64, 32))
    cv2.imwrite(source_dir, image)


if __name__ == "__main__":
    workbook = createXls()
    write2Xls(fileDir, workbook)
    workbook.save('鞋印数据统计.xls')