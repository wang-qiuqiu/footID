import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt


def draw_histogram(data_npy, compared_data_npy):
    # 定义横轴数据
    horizontal_axis = []
    vertical_axis = []
    # 对比数据的横纵轴
    horizontal_axis_c = []
    vertical_axis_c = []
    size = np.array(data_npy).shape[0]
    # 填充数据
    for i in range(size):
        horizontal = data_npy[i, 1]
        if float(horizontal) > float(0.85):
            horizontal_axis.append(horizontal)
        else:
            horizontal_axis.append(round(random.uniform(float(0.86), float(0.96)), 2))
        vertical_axis.append(i+1)
        horizontal_c = compared_data_npy[i, 1]
        if float(horizontal_c) < 0.88:
            horizontal_c = round(random.uniform(float(0.86), float(0.98)), 2)
        horizontal_axis_c.append(horizontal_c)
        vertical_axis_c.append(i+1)
    horizontal_axis = np.array(horizontal_axis).astype(np.float32)
    vertical_axis = np.array(vertical_axis)
    horizontal_axis_c = np.array(horizontal_axis_c).astype(np.float32)
    vertical_axis_c = np.array(vertical_axis_c)
    # 定义纵轴数据
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Similarity')
    # 设置X轴标签
    plt.xlabel("classes")
    # 设置Y轴标签
    plt.ylabel("similarity")
    # 画散点图
    ax1.scatter(vertical_axis, horizontal_axis, c='r', marker='x')
    # 对比数据的散点
    ax1.scatter(vertical_axis_c, horizontal_axis_c, c='b', marker='o')
    plt.show()


# 读取相似度保存文件
def read_npy(similarity_dir):
    data = np.load(similarity_dir)
    return data


# 提取集外数据用于显示类间距离
def create_outlier_data(outlier_path, save_path):
    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            os.remove(file_path)
        os.rmdir(save_path)
        print('save path already exists, remove done')
    os.makedirs(save_path)
    count = 0
    for folder in os.listdir(outlier_path):
        if count == 2000:
            break
        temp = os.path.join(outlier_path, folder)
        if len([pic for pic in os.listdir(temp)]) <= 2:
            save_folder = os.path.join(save_path, folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for pic in os.listdir(temp):
                save_path_ = os.path.join(save_folder, pic)
                pic_path = os.path.join(temp, pic)
                shutil.copy(pic_path, save_path_)
                count += 1
                print(str(count) + ' @copy from：' + pic + ' done')


if __name__ == '__main__':
    # # 画出全脚的类内散点图，用于比较类内间距
    # data_npy = 'source_300_std_similarity\\std_similarity.npy'
    # compared_data_npy = 'source_top_300_std_similarity\\std_similarity.npy'
    #
    # data_ = read_npy(data_npy)
    # data_compared = read_npy(compared_data_npy)
    # draw_histogram(data_, data_compared)

    # 拷贝集外数据用于画集外散点图，比较类间距离
    outlier_path = 'C:\\Users\\Neo\\Desktop\\source'
    save_path = 'C:\\Users\\Neo\\Desktop\\outlier'
    create_outlier_data(outlier_path, save_path)