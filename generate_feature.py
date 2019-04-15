from footid1 import *
import numpy as np
import os
import cv2


def generate_feature(pic_dir, feature_dir):
    feature_list = []
    i = 0
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        for folder in os.listdir(pic_dir):
            temp_dir = os.path.join(pic_dir, folder)
            for pic in os.listdir(temp_dir):
                list_for_data = []
                train_pic_path = os.path.join(temp_dir, pic)
                train_raw_img = cv2.imread(train_pic_path)
                list_for_data.append(train_raw_img)
                nparray_for_data = np.array(list_for_data)
                pic_feature = sess.run(h5, {h0: nparray_for_data})
                i += 1
                print(str(i) + ' @生成图片特征：' + str(pic) + ' 完成!')
                feature_list.append(pic_feature)
    feature_nparray = np.array(feature_list)
    # 生成特征存放文件夹，若文件夹存在则全部删除
    if os.path.exists(feature_dir):
        for feature in os.listdir(feature_dir):
            temp_dir = os.path.join(feature_dir, feature)
            os.remove(temp_dir)
        os.rmdir(feature_dir)
    os.makedirs(feature_dir)
    print('-----------------')
    print('所有图片特征生成完成!')
    print('特征保存路径：' + os.path.join(os.getcwd(), feature_dir))
    np.save(feature_dir + '\\data_feature', feature_nparray)


if __name__ == '__main__':
    # 读取所有训练图生成160维的特征
    train_dir = 'C:\\Users\\Neo\\Desktop\\source_300_train'
    train_feature_dir = 'display_300_feature'
    generate_feature(train_dir, train_feature_dir)

    # # 生成所有集外测试图的160维特征
    # test_dir = 'C:\\Users\\JJD2WY\\Desktop\\test'
    # test_feature_dir = 'test_data_feature'
    # generate_feature(test_dir, test_feature_dir)
    # npy_dir = 'source_300_top_feature\\data_feature.npy'
    # a = np.load(npy_dir)
    #
    # npy_dir_ = 'source_300_feature\\data_feature.npy'
    # b = np.load(npy_dir_)
    # print()