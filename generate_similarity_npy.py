from footid1 import *
import cv2
import os
import numpy as np
from predict_foot import cosine


def generate_similarity(std_path):
    folder_list = []
    similarity_list = []

    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        for folder in os.listdir(std_path):
            folder_feature_list = []
            folder_list.append(folder)
            temp = os.path.join(std_path, folder)
            for pic in os.listdir(temp):
                img_list = []
                temp_ = os.path.join(temp, pic)
                img = cv2.imread(temp_)
                img_list.append(img)
                img = np.array(img_list)
                feature = sess.run(h5, {h0: img})
                folder_feature_list.append(feature)
            similarity_mean = compute_mean_similarity(folder_feature_list)
            similarity_list.append(similarity_mean)
            print('文件夹：' + folder + ' 相似度标准值生成完成！')
        sess.close()
    target_folder = 'display_std_similarity'
    if os.path.exists(target_folder):
        for file in os.listdir(target_folder):
            temp_dir = os.path.join(target_folder, file)
            os.remove(temp_dir)
        os.rmdir(target_folder)
    os.makedirs(target_folder)
    result = np.array([folder_list, similarity_list])
    result = result.transpose()
    np.save(target_folder + '\\std_similarity', result)


# 计算一个文件夹内相似度均值
def compute_mean_similarity(feature_list):
    result_list = []
    for i in range(len(feature_list)-1):
        for j in range(i, len(feature_list)):
            result = cosine(feature_list[i], feature_list[j])
            result_list.append(result)
    feature_array = np.array(result_list)
    feature_array = np.reshape(feature_array, (feature_array.shape[0], feature_array.shape[1]))
    mean = np.mean(feature_array)
    return mean


if __name__ == '__main__':
    std_path = 'C:\\Users\\Neo\\Desktop\\source_300_train'
    generate_similarity(std_path)