import os
import shutil
from footid1 import *
from scipy import ndimage
import scipy


# 选择集外图片
def pick_outlier():
    count = 0
    std_path = 'C:\\Users\\Neo\\Desktop\\source_test'
    source_path = 'C:\\Users\\Neo\\Desktop\\source'
    target_path = 'C:\\Users\\Neo\\Desktop\\display_outlier'

    deleteFolder(target_path)

    std_folder = []
    for folder in os.listdir(std_path):
        std_folder.append(folder)

    for folder in os.listdir(source_path):
        if count == 1000:
            break
        if folder in std_folder:
            continue
        else:
            temp_path = os.path.join(source_path, folder)
            for pic in os.listdir(temp_path):
                pic_path = os.path.join(temp_path, pic)
                shutil.copy(pic_path, target_path)
                count += 1
                print("拷贝：" + str(count) + " " + pic + "完成")
                break


# 选图(集外)
def select_outlier():
    source_path = 'C:\\Users\\Neo\\Desktop\\display_outlier'
    lis = np.load('display_data_label_npy\\data_label.npy')
    source = np.load('display_std_similarity\\std_similarity.npy')
    count = 0
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        for pic in os.listdir(source_path):
            pic_path = os.path.join(source_path, pic)
            image = np.array(ndimage.imread(pic_path, flatten=False))
            image = scipy.misc.imresize(image, size=(64, 32))
            cv2.imwrite('./pick.jpg', image)
            img = cv2.imread('./pick.jpg')
            img_list = []
            img_list.append(img)
            img = np.array(img_list)
            result_folder = ''
            h_test_predict = sess.run(tf.nn.softmax(y), {h0: img})
            class_ = tf.argmax(h_test_predict, 1)
            class_ = sess.run(class_)
            for i in range(lis.shape[0]):
                if lis[i, 1] == str(class_[0]):
                    result_folder = lis[i, 0]
                    break
            target = ''
            for row in range(source.shape[0]):
                if source[row, 0] == result_folder:
                    target = source[row, 1]
            t = float(0.90)
            if float(target) < t:
                count += 1
                print(str(count) + '可选：' + '' + pic)
            else:
                os.remove(pic_path)


# 选图(集内)
def select_inner():
    source_path = 'C:\\Users\\Neo\\Desktop\\display_inner'
    lis = np.load('display_data_label_npy\\data_label.npy')
    source = np.load('display_std_similarity\\std_similarity.npy')
    count = 0
    with tf.Session() as sess:
        foot_saver.restore(sess, 'display_checkpoint\\30000.ckpt')
        for pic in os.listdir(source_path):
            pic_path = os.path.join(source_path, pic)
            image = np.array(ndimage.imread(pic_path, flatten=False))
            image = scipy.misc.imresize(image, size=(64, 32))
            cv2.imwrite('./pick.jpg', image)
            img = cv2.imread('./pick.jpg')
            img_list = []
            img_list.append(img)
            img = np.array(img_list)
            result_folder = ''
            h_test_predict = sess.run(tf.nn.softmax(y), {h0: img})
            class_ = tf.argmax(h_test_predict, 1)
            class_ = sess.run(class_)
            for i in range(lis.shape[0]):
                if lis[i, 1] == str(class_[0]):
                    result_folder = lis[i, 0]
                    break
            target = ''
            for row in range(source.shape[0]):
                if source[row, 0] == result_folder:
                    target = source[row, 1]
            t = float(0.90)
            if float(target) < t:
                os.remove(pic_path)
            else:
                count += 1
                print(str(count) + '可选：' + '' + pic)


# 清空文件夹
def deleteFolder(target_path):
    if os.path.exists(target_path):
        if len(os.listdir(target_path)) > 0:
            for file in os.listdir(target_path):
                os.remove(os.path.join(target_path, file))
            os.rmdir(target_path)
        else:
            os.rmdir(target_path)
    os.makedirs(target_path)


# 选择集内图片
def pick_inner():
    count_all = 0
    pick_path = 'C:\\Users\\Neo\\Desktop\\source_test'
    target_path = 'C:\\Users\\Neo\\Desktop\\display_inner'

    deleteFolder(target_path)

    for folder in os.listdir(pick_path):
        # if count_all == 500:
        #     break
        folder_path = os.path.join(pick_path, folder)
        count = 0
        for pic in os.listdir(folder_path):
            if count == 3:
                break
            else:
                pic_path = os.path.join(folder_path, pic)
                shutil.copy(pic_path, target_path)
                count += 1
                count_all += 1
                print("拷贝：" + str(count_all) + " " + pic + "完成")


def main():
    # pick_outlier()
    pick_inner()
    # select_outlier()
    # select_inner()


if __name__ == '__main__':
    main()