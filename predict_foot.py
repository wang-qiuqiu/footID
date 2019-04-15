from footid1 import *
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import scipy
from PIL import Image

standard_inner_data_path = 'C:\\Users\\JJD2WY\\Desktop\\train'


# 相似度计算：余弦距离
def cosine(a, b):
    t = np.dot(a, b.T)
    k = np.linalg.norm(a) * np.linalg.norm(b)
    cos = t / k
    return cos


# 计较两张图的相似度
def compute_distance_one(standard_path, test_path):
    # 读取图片
    img = cv2.imread(test_path)
    standard_img = cv2.imread(standard_path)

    # 转换成需要的输入格式(?, 64, 32, 3)
    img_list = []
    img_list.append(img)
    img = np.array(img_list)

    standard_img_list = []
    standard_img_list.append(standard_img)
    standard_img = np.array(standard_img_list)

    # 载入模型
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')

        # 得到两张图的160维向量
        h_test_predict = sess.run(h5, {h0: img})
        standard_h_test_predict = sess.run(h5, {h0: standard_img})

        # 由余弦距离比较相似度
        pre_result = cosine(h_test_predict, standard_h_test_predict)
        # pre_result = np.array(cosine(h_test_predict, standard_h_test_predict))

        print(u"预测：", pre_result)
        sess.close()


# 预测单张图的所属类别
def predict_one_img_class(img_path):
    img = cv2.imread(img_path)
    img_list = []
    img_list.append(img)
    img = np.array(img_list)
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        h_test_predict = sess.run(tf.nn.softmax(y), {h0: img})
        class_ = tf.argmax(h_test_predict, 1)
        class_ = sess.run(class_)
        acc = h_test_predict[0, class_]
        # 画出概率分布直方图
        draw_histogram(h_test_predict, class_num=190)
        # 加载数据标签的对应文件输出类别
        lis = np.load('data_label_npy\\data_label.npy')
        for i in range(lis.shape[0]):
            if lis[i, 1] == str(class_[0]):
                print("所属类别为：" + str(lis[i, 0]))
                print("top1: " + str(acc[0]))
                break


# 集外所有图片预测outlier_path为图片根路径
def predict_outlier_batch(outlier_path):
    # 加载data_label文件用于错误预测图片查找错判类别
    lis = np.load('data_label_300_full_npy\\data_label.npy')
    outlier_count = 0
    outlier_correct = 0
    outlier_error_predict_list = []
    with tf.Session() as sess:
        for file in os.listdir(outlier_path):
            temp_dir = os.path.join(os.path.join(outlier_path, file))
            for outlier_pic in os.listdir(temp_dir):
                outlier_list = []
                outlier = cv2.imread(os.path.join(temp_dir, outlier_pic))
                outlier_list.append(outlier)
                outlier_count += 1
                outlier_array = np.array(outlier_list)
                foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
                outlier_predict = sess.run(tf.nn.softmax(y), {h0: outlier_array})
                class_ = tf.argmax(outlier_predict, 1)
                class_ = sess.run(class_)
                acc = outlier_predict[0, class_]
                print("预测图片：" + str(outlier_pic) + " --> " + str(acc[0]))
                # 模型对于集内预测率很高，将近100%，此处将阈值设为0.9。预测概率小于0.9则为集外
                if acc[0] < 0.9:    # 集外
                    outlier_correct += 1
                else:   # 错判成集内
                    for i in range(lis.shape[0]):
                        if lis[i, 1] == str(class_[0]):
                            outlier_error_predict_list.append(os.path.join(temp_dir, outlier_pic)
                                                              + ' --> ' + str(lis[i, 0]))
        acc_outlier = outlier_correct/outlier_count
        print('集外总数为：' + str(outlier_count) + '张')
        print('正确预测集外：' + str(outlier_correct) + '张')
        print("正确率：" + str(acc_outlier))

        # 将预测错误图片路径保存，生成npy文件，便于查看
        outlier_error_predict_nparray = np.array(outlier_error_predict_list)
        if not os.path.exists('outlier_error_predict'):
            os.makedirs('outlier_error_predict')
        np.save('outlier_error_predict\\error_predict_img', outlier_error_predict_nparray)


def show_outlier_error_predict(errorpath):
    error_nparray = np.load(errorpath)
    for img_path in error_nparray:
        print(img_path)


def draw_histogram(prob, class_num):
    # 定义横轴数据
    horizontal_axis = []
    for i in range(np.array(prob).shape[1]):
        horizontal_axis.append(prob[0, i])
    # 定义纵轴数据
    # vertical_axis = [i for i in range(class_num)]
    plt.bar(range(len(horizontal_axis)), horizontal_axis, tick_label='')
    plt.xlabel("classes")
    plt.ylabel("prob")
    plt.title("Probability distributions")
    plt.show()


# 测试图与预测集内所有图片计算相似度
def compute_distance_in_class(target_img_path):
    correct_count = 0
    img = cv2.imread(target_img_path)
    img_list = []
    img_list.append(img)
    img = np.array(img_list)
    # 目标文件夹
    result_folder = ''
    # 测试图特征
    # test_feature = np.array([])
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        h_test_predict = sess.run(tf.nn.softmax(y), {h0: img})
        feature = sess.run(h5, {h0: img})
        test_feature = feature
        class_ = tf.argmax(h_test_predict, 1)
        class_ = sess.run(class_)
        # 加载数据标签的对应文件输出类别
        lis = np.load('display_data_label_npy\\data_label.npy')
        for i in range(lis.shape[0]):
            if lis[i, 1] == str(class_[0]):
                result_folder = lis[i, 0]
                break
        sess.close()

    # 加载源文件
    source = np.load('display_std_similarity\\std_similarity.npy')
    target = ''
    for row in range(source.shape[0]):
        if source[row, 0] == result_folder:
            target = source[row, 1]

    cosine_list = []
    folder = os.path.join(standard_inner_data_path, result_folder)
    feature_list = generate_feature_from_folder(folder)
    for std_feature in feature_list:
        cosine_list.append(cosine(std_feature, test_feature))
    cosine_array = np.array(cosine_list)
    cosine_array = np.reshape(cosine_array, (cosine_array.shape[0], cosine_array.shape[1]))
    mean = np.mean(cosine_array)
    t = np.float32(target)

    if mean < t:
        print('预测为集外')
        print('相似度为：' + str(mean) + ' @标准相似度为：' + str(t))
        # correct_count = 1
        return False
    else:
        print('预测为集内')
        print('相似度为：' + str(mean) + ' @标准相似度为：' + str(t))
        print('预测类别为：' + result_folder)
        return result_folder
    # return correct_count


# 计算指定文件夹内所有图片的特征
def generate_feature_from_folder(folder_path):
    feature_list = []
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        for img in os.listdir(folder_path):
            img_list = []
            temp_dir = os.path.join(folder_path, img)
            source_img = cv2.imread(temp_dir)
            img_list.append(source_img)
            source_img = np.array(img_list)
            feature = sess.run(h5, {h0: source_img})
            feature_list.append(feature)
        return feature_list


def predict_all_outlier(outlier_path):
    pic_count = 0
    correct_count = 0
    for folder in os.listdir(outlier_path):
        temp = os.path.join(outlier_path, folder)
        for pic in os.listdir(temp):
            pic_count += 1
            pic_path = os.path.join(temp, pic)
            count = compute_distance_in_class(pic_path)
            if count == 1:
                correct_count += 1
    print("集外图片总数为：" + str(pic_count))
    print("正确预测集外图片：" + str(correct_count))
    print("集外预测率为：" + str(correct_count/pic_count*100) + "%")


# 如果是原始图，则resize到64*32
def resize(dir):
    image = np.array(ndimage.imread(dir, flatten=False))
    image = scipy.misc.imresize(image, size=(64, 32))
    os.remove(dir)
    cv2.imwrite(dir, image)


if __name__ == '__main__':
    # # 单张图的预测
    # img_path = 'C:\\Users\\Neo\\Desktop\\6.jpg'
    # resize(img_path)
    # predict_one_img_class(img_path)

    # 批量预测集外图片
    # outlier_path = 'C:\\Users\\JJD2WY\\Desktop\\test'
    # predict_outlier_batch(outlier_path)

    # # 控制台显示所有预测错误集外图片名称
    # error_path = 'outlier_error_predict\\error_predict_img.npy'
    # show_outlier_error_predict(error_path)

    # # 计算两张图的相似度
    # test_path = 'C:\\Users\\JJD2WY\\Desktop\\train\\C151224000129\\S1512240001290004D.jpg'
    # standard_path = 'C:\\Users\\JJD2WY\\Desktop\\train\\C151224000006\\P1512240000060010D.jpg'
    # compute_distance_one(standard_path, test_path)

    # # 计算预测图片与目标文件夹中所有样本的相似度
    target_img_path = 'C:\\Users\\Neo\\Desktop\\source\\1_115\\008530_imageR410142000011201211002201.jpg'
    ########
    full = Image.open(target_img_path)
    full_show = full.resize((32, 64))
    full_show = np.asarray(full_show)
    cv2.imwrite('test.jpg', full_show)
    ########
    compute_distance_in_class('test.jpg')

    # #计算所有outlier的准确
    # outlier_path = 'C:\\Users\\JJD2WY\\Desktop\\test_new'
    # predict_all_outlier(outlier_path)