from footid1 import *
from crop_data import crop2top, crop2bottom
from tkinter import *
import tkinter.filedialog
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

root_ = 'C:\\Users\\Neo\\Desktop\\source_test'
test = 'test.jpg'
standard_inner_data_path = 'C:\\Users\\Neo\\Desktop\\source_300_train'


def main(path):
    show_raw_data(path)
    show_4_scale(path)
    flag = predict(test)
    if flag is not False:
        status.set(1)
        result.set('结果是：'+flag)
        show_result(flag)
    else:
        result.set('结果是：集外未知类别')


# 从GUI窗口中获取待检测图片路径
def save_path():
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        lb.config(text="您选择的文件是：" + filename)
        main(filename)
    else:
        lb.config(text="您没有选择任何文件")
    return


# 显示输入图像
def show_raw_data(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow('RawImage')
    cv2.imshow('RawImage', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 用于展示四个尺度
def show_4_scale(img_path):
    # 全脚
    full = Image.open(img_path)
    full_show = np.asarray(full)
    # 脚掌
    top = crop2top(img_path)
    top_show = top.resize((32, 64))
    top_show = np.asarray(top_show)
    # 脚跟
    bottom = crop2bottom(img_path)
    bottom_show = bottom.resize((32, 64))
    bottom_show = np.asarray(bottom_show)
    # 脚掌中心
    center = get_center(top)
    center_show = center.resize((32, 64))
    center_show = np.asarray(center_show)
    # 合并后在一个窗口显示
    img_show = np.hstack([full_show, top_show, bottom_show, center_show])
    cv2.namedWindow('Multi-scale')
    cv2.imshow('Multi-scale', img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pic_name = 'test.jpg'
    if os.path.exists(pic_name):
        os.remove(pic_name)
    cv2.imwrite(pic_name, full_show)


# 获取脚掌中心
def get_center(im):
    x_size, y_size = im.size
    start_point_xy = x_size / 4
    end_point_xy = x_size / 4 + x_size / 2
    start_point_yx = y_size / 4
    end_point_yx = y_size / 4 + y_size / 2
    box = (start_point_xy, start_point_yx, end_point_xy, end_point_yx)
    new_im = im.crop(box)
    return new_im


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


# 预测结果
def predict(img_path):
    img = cv2.imread(img_path)
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
    # 阈值
    t = float(0.80)
    if float(target) < t:
        print('预测为集外')
        return False
    else:
        print('预测为集内')
        print('预测类别为：' + result_folder)
        return result_folder


# 显示预测结果对应文件夹中所有图片
def show_result(path):
    resultShow = np.zeros((128, 64))
    _path = os.path.join(root_, path)
    for pic in os.listdir(_path):
        pic_path = os.path.join(_path, pic)
        pic_raw = Image.open(pic_path)
        # 统一尺寸
        pic_resize = pic_raw.resize((64, 128))
        pic_array = np.asarray(pic_resize)
        if len(pic_array.shape) > 2:
            pic_array = cv2.cvtColor(pic_array, cv2.COLOR_RGB2GRAY)
        resultShow = np.hstack([resultShow, pic_array])
    resultShow = resultShow[:, 65:]
    cv2.namedWindow('Result')
    cv2.imshow('Result', resultShow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 选择测试图片并调用主函数
def chooseFile():
    filename = tkinter.filedialog.askopenfilename(title='选择文件')
    e.set(filename)


if __name__ == '__main__':
    root = Tk()
    root.title('鞋印图像开集分类')

    lb = Label(root, text='请选择待测试图像')
    lb.pack()

    # 设置变量
    e = StringVar()
    result = StringVar()
    status = IntVar()

    result.set('结果是：')
    e_entry = Entry(root, width=68, textvariable=e)
    e_entry.pack()

    submit_button = Button(root, text="选择", command=chooseFile)
    submit_button.pack()
    classify_button = Button(root, text="分类", command=lambda: main(e.get()))
    classify_button.pack()

    result_label = Label(root, textvariable=result)
    result_label.pack()

    root.mainloop()