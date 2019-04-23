from footid1 import *
from sklearn.metrics import auc
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random

X_positive = np.load('display_train\\TestX.npy')
# positive_data_result = create_positive_data()
Y_positive = np.load('display_train\\TestY.npy')


def load_negative_data(negative_path):
    result_list = []
    for folder in os.listdir(negative_path):
        temp = os.path.join(negative_path, folder)
        for img_path in os.listdir(temp):
            temp_ = os.path.join(temp, img_path)
            img = cv2.imread(temp_)
            print("读取图像：" + temp_)
            result_list.append(img)
    negative_data_result = np.array(result_list)
    print("--> 读取所有负样本完成！")
    return negative_data_result


def load_positive_data(positive_path):
    result_list = []
    for folder in os.listdir(positive_path):
        temp = os.path.join(positive_path, folder)
        for img_path in os.listdir(temp):
            temp_ = os.path.join(temp, img_path)
            img = cv2.imread(temp_)
            print("读取图像：" + temp_)
            result_list.append(img)
        positiv_data_result = np.array(result_list)
    print("--> 读取所有正样本完成！")
    return positiv_data_result


def create_data(positive_data_result, negative_data_result):
    # positive_data_result = positive_data_result[:100, :, :, :]
    x_test_list = []
    x_test_ = []
    for i in range(positive_data_result.shape[0]):
        x_test_list.append(positive_data_result[i, :, :, :])
    for i in range(negative_data_result.shape[0]):
        x_test_list.append(negative_data_result[i, :, :, :])
    x_test = np.array(x_test_list)
    y_test_list = []
    for i in range(x_test.shape[0]):
        if i < positive_data_result.shape[0]:
            y_test_list.append(1)
        else:
            y_test_list.append(0)
    y_test_list = [int(i) for i in y_test_list]
    y_test = np.array(y_test_list)
    index = [i for i in range(len(y_test_list))]
    index = np.array(index)
    temp = np.vstack((index, y_test))
    temp = temp.transpose()
    np.random.shuffle(temp)
    x_test_index = temp[:, 0]
    for index_ in x_test_index:
        target = x_test_list[index_]
        x_test_.append(target)
    x_test = np.array(x_test_)
    y_test = temp[:, 1]
    return x_test, y_test


def get_score(x_test):
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        y_scores = []
        for i in range(x_test.shape[0]):
            img_temp = []
            img_pre = x_test[i, :, :, :]
            img_temp.append(img_pre)
            img_pre = np.array(img_temp)
            scores = sess.run(tf.nn.softmax(y), {h0: img_pre})
            index = tf.argmax(scores, 1)
            index = sess.run(index)
            index = index[0]
            score = scores[0, index]
            print("--> 预测图片：" + str(i) + '完成')
            y_scores.append(score)
        sess.close()
        y_scores = np.array(y_scores).astype(np.float32)
        return y_scores


# 计算AUC并画出ROC曲线
def create_roc(y_test, y_score):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return


def save_y_scores(save_path, y_scores, y_test):
    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            temp = os.path.join(save_path, file)
            os.remove(temp)
        os.rmdir(save_path)
    os.makedirs(save_path)
    np.save(save_path + '\\y_scores', y_scores)
    np.save(save_path + '\\y_test', y_test)
    print('模型预测得分保存完成！')


def create_positive_data():
    data = np.load('display_train\\TrainX.npy')
    rand = []
    result_list = []
    for index in range(800):
        ran = random.randint(0, data.shape[0]-1)
        rand.append(ran)
    rand = [int(i) for i in rand]
    for index in rand:
        img_path = data[index]
        img = cv2.imread(img_path)
        result_list.append(img)
    x_test = np.array(result_list)
    return x_test


if __name__ == '__main__':
    # # 加载数据
    negative_path = 'C:\\Users\\Neo\\Desktop\\roc_display_outlier'
    negative_data_result = load_negative_data(negative_path)

    positive_path = 'C:\\Users\\Neo\\Desktop\\roc_display_inner'
    positive_data_result = load_positive_data(positive_path)

    X_test, Y_test = create_data(positive_data_result, negative_data_result)
    Y_scores = get_score(X_test)
    Y_scores_save_path = 'roc_y_scores'
    save_y_scores(Y_scores_save_path, Y_scores, Y_test)
    y_score = np.load('roc_y_scores\\y_scores.npy')
    y_test = np.load('roc_y_scores\\y_test.npy')
    create_roc(y_test, y_score)
