import math
import os
import numpy as np
import cv2

# class_num = 54
class_num = 300

# 数据和标签的npy文件存储位置
npy_dir = 'display_data_label_npy'


# 生成训练集和测试集数据
def get_file(file_dir):
    # step1：获取路径下所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中。
    data_list = [[] for i in range(class_num)]
    label_list_ = [[] for i in range(class_num)]

    # 数据和标签添加到npz文件便于观察
    store_data_list = []
    store_label_list = []

    folder_count = 0
    class_count = 0
    i = 0
    for folder in os.listdir(file_dir):
        for pic in os.listdir(os.path.join(file_dir, folder)):
            pic_dir = os.path.join(file_dir, folder)
            pic_dir = os.path.join(pic_dir, pic)
            data_list[folder_count].append(pic_dir)
            print('读取鞋印图像：' + pic_dir + ' 完成')
            i += 1
            label_list_[folder_count].append(class_count)
        store_data_list.append(folder)
        store_label_list.append(class_count)
        class_count += 1
        folder_count += 1

    print('共有鞋印图像：' + str(i) + '张')
    # 数据和标签生成npz文件用于后续观察
    create_npz(store_data_list, store_label_list)

    # step2：对生成的图片路径和标签List做打乱处理把所有的合起来组成一个list（img和lab）
    # 合并数据numpy.hstack(tup)
    # tup可以是python中的元组（tuple）、列表（list），或者numpy中数组（array），函数作用是将tup在水平方向上（按列顺序）合并
    image_list = np.array([])
    label_list = np.array([])
    for lis in data_list:
        image_list = np.hstack((image_list, lis))
    for lab in label_list_:
        label_list = np.hstack((label_list, lab))
    # 利用shuffle，转置、随机打乱
    temp = np.array([image_list, label_list])  # 转换成2维矩阵
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)  # 按行随机打乱顺序函数

    # 将所有的img和lab转换成list
    all_image_list = temp[:, 0] # 取出第0列数据，即图片路径
    all_label_list = list(temp[:, 1])  # 取出第1列数据，即图片标签
    all_label_list = [int(float(i)) for i in all_label_list]  # 转换成int数据类型

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    ratio = 0.2
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数, ratio是测试集的比例
    n_train = n_sample - n_val  # 训练样本数
    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]  # 转换成int数据类型
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]  # 转换成int数据类型

    val_data = val_images
    val_label = val_labels

    val_data_list = []
    val_label_batch = np.zeros((len(val_label), class_num))

    val_batch_index = 0
    for dir in val_data:
        val_img = cv2.imread(dir)
        val_img = np.array(val_img).astype(np.float32)
        val_data_list.append(val_img)
        val_label_batch[val_batch_index][val_label[val_batch_index]] = 1
        val_batch_index += 1

    val_data_batch = np.array(val_data_list)

    save_path = 'display_train'

    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            temp_path = os.path.join(save_path, file)
            os.remove(temp_path)
        os.rmdir(save_path)
    os.makedirs(save_path)
    
    np.save(save_path + '\\TrainX', tra_images)
    np.save(save_path + '\\TrainY', tra_labels)
    np.save(save_path + '\\TestX', val_data_batch)
    np.save(save_path + '\\TestY', val_label_batch)
    print('训练集、测试集生成！')
    # return tra_images, tra_labels, val_images, val_labels
    # return all_image_list, all_label_list


def create_npz(store_data_list, store_label_list):
    temp = np.array([store_data_list, store_label_list])
    temp = temp.transpose()
    if os.path.exists(npy_dir):
        for file in os.listdir(npy_dir):
            os.remove(os.path.join(npy_dir, file))
    else:
        os.makedirs(npy_dir)
    np.save(npy_dir + '/data_label', temp)
    return


def read_npz(npy_file):
    data = np.load(npy_dir + '/data_label.npy')
    print(data)
    return


if __name__ == '__main__':
    footprint_datadir = 'C:\\Users\\Neo\\Desktop\\source_300_train'
    # footprint_datadir = 'C:\\Users\\LY2JJD\\Desktop\\tensorflow-vgg-master\\footprint_exp\\data\\new'
    get_file(footprint_datadir)
    # read_npz(npy_dir)
    # result = np.load('foot_data_train/TrainY.npy')
    # print()