import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2
import random


def cosine(a, b):
    t = np.dot(a, b.T)
    k = np.linalg.norm(a) * np.linalg.norm(b)
    cos = t / k
    return cos


# 提取特征
def gen_feature(pic_dir, save_path):
    # 开启图用于恢复网络
    g = tf.Graph()
    sess = tf.Session(graph=g)
    with sess.as_default():
        with g.as_default():
            # # 全脚的
            # saver = tf.train.import_meta_graph('source_300_checkpoint\\50000.ckpt.meta')
            # saver.restore(sess, 'source_300_checkpoint\\50000.ckpt')
            # 脚掌的
            saver = tf.train.import_meta_graph('source_300_top_checkpoint\\30000.ckpt.meta')
            saver.restore(sess, 'source_300_top_checkpoint\\30000.ckpt')
    clear_folder(save_path)
    i = 0
    with sess.as_default():
        with sess.graph.as_default():
            graph = tf.get_default_graph()
            h0 = graph.get_tensor_by_name('input/x:0')
            h5 = tf.get_collection('foot_feature')[0]
            for folder in os.listdir(pic_dir):
                feature_list = []
                temp_dir = os.path.join(pic_dir, folder)
                save_name = save_path + '\\' + folder
                for pic in os.listdir(temp_dir):
                    list_for_data = []
                    train_pic_path = os.path.join(temp_dir, pic)
                    train_raw_img = cv2.imread(train_pic_path)
                    list_for_data.append(train_raw_img)
                    nparray_for_data = np.array(list_for_data)
                    pic_feature = sess.run(h5, {h0: nparray_for_data})
                    feature_list.append(pic_feature)
                feature_from_folder = np.array(feature_list)
                np.save(save_name, feature_from_folder)
                i += 1
                print(str(i) + ' generate feature from：' + folder + ' done')
    sess.close()


# 清空并重新建文件夹
def clear_folder(save_path):
    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            os.remove(file_path)
        os.rmdir(save_path)
    os.makedirs(save_path)


# 计算一个文件夹内的平均特征
def compute_mean_feature(feature_npy):
    feature_all = np.load(feature_npy)
    size = feature_all.shape[0]
    result = feature_all[0, :, :]
    for i in range(size-1):
        result = np.vstack((result, feature_all[i+1, :, :]))
    result_mean = np.mean(result, 0)
    result_mean = np.reshape(result_mean, (1, result_mean.shape[0]))
    return result_mean


# 每个类都计算个平均特征
def walk_through_folder(path):
    feature_list = []
    for file in os.listdir(path):
        file_name = os.path.join(path, file)
        mean_feature = compute_mean_feature(file_name)
        feature_list.append(mean_feature)
    return np.array(feature_list)


def cal_similarity(feature_all):
    score = []
    size = feature_all.shape[0]
    for i in range(size):
        score_min = 0
        for j in range(size):
            if i == j:
                continue
            else:
                score_ = cosine(feature_all[i, :, :], feature_all[j, :, :])
                if score_ > score_min:
                    score_min = score_
        score.append(float(score_min))
    return score


# 画出散点图
def draw_histogram(data_list, compared_data_list):
    vertical_axis = [i+1 for i in range(300)]
    vertical_axis_c = [i+1 for i in range(300)]
    compare_data = []
    for data in compared_data_list:
        if float(data) > float(0.625):
            # data = round(random.uniform(float(0.45), float(0.66)), 2)
            compare_data.append(data)
        else:
            compare_data.append(data)
    horizontal_axis = np.array(data_list).astype(np.float32)
    vertical_axis = np.array(vertical_axis)
    horizontal_axis_c = np.array(compare_data).astype(np.float32)
    vertical_axis_c = np.array(vertical_axis_c)
    # 定义纵轴数据
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Similarity Between Classes')
    # 设置X轴标签
    plt.xlabel("classes")
    # 设置Y轴标签
    plt.ylabel("similarity")
    # 画散点图
    ax1.scatter(vertical_axis, horizontal_axis, c='r', marker='x')
    # 对比数据的散点
    ax1.scatter(vertical_axis_c, horizontal_axis_c, c='b', marker='o')
    plt.show()


if __name__ == '__main__':
    # # 生成全脚的特征
    # source = 'C:\\Users\\Neo\\Desktop\\source_300'
    # target = 'C:\\Users\\Neo\\Desktop\\source_300_feature'
    # gen_feature(source, target)

    # 生成脚掌的特征
    # source = 'C:\\Users\\Neo\\Desktop\\source_top_300'
    # target = 'C:\\Users\\Neo\\Desktop\\source_top_300_feature'
    # gen_feature(source, target)

    path = 'C:\\Users\\Neo\\Desktop\\source_300_feature'
    feature_ = walk_through_folder(path)
    s = cal_similarity(feature_)

    path_ = 'C:\\Users\\Neo\\Desktop\\source_top_300_feature'
    feature_1 = walk_through_folder(path_)
    s_ = cal_similarity(feature_1)
    draw_histogram(s, s_)
    print()