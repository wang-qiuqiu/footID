from PIL import Image
from scipy.ndimage import filters
import os
import numpy as np
import cv2


# 使用不同方式处理图片
def increase_data(source_path, target_path):
    i = 0
    for folder in os.listdir(source_path):
        temp_folder = os.path.join(source_path, folder)
        for pic in os.listdir(temp_folder):
            pic_path = os.path.join(temp_folder, pic)
            img = Image.open(pic_path)

            # 左旋转10
            img_rotate_10 = img.rotate(5, expand=False)
            name = pic.split('.')[0] + '_rotate10.jpg'
            save(img_rotate_10, name, folder, target_path)

            img_rotate_10_flip = img_rotate_10.transpose(Image.FLIP_LEFT_RIGHT)
            name = pic.split('.')[0] + '_rotate_10_flip.jpg'
            save(img_rotate_10_flip, name, folder, target_path)

            # 右旋转10
            img_rotate_350 = img.rotate(355, expand=False)
            name = pic.split('.')[0] + '_rotate350.jpg'
            save(img_rotate_350, name, folder, target_path)

            img_rotate_350_flip = img_rotate_350.transpose(Image.FLIP_LEFT_RIGHT)
            name = pic.split('.')[0] + '_rotate_350_flip.jpg'
            save(img_rotate_350_flip, name, folder, target_path)

            # 高斯噪声
            img_cv2 = cv2.imread(pic_path)
            # img_awn = AWN_image(img_cv2, 0.1)
            # img_awn = Image.fromarray(np.uint8(img_awn))
            # name = pic.split('.')[0] + '_g_noise.jpg'
            # save(img_awn, name, folder, target_path)
            #
            # img_awn_flip = img_awn.transpose(Image.FLIP_LEFT_RIGHT)
            # name = pic.split('.')[0] + '_g_noise0_flip.jpg'
            # save(img_awn_flip, name, folder, target_path)

            # 椒盐噪声
            img_aspn = ASPN_image(img_cv2, 0.9)
            img_aspn = img_aspn.transpose(2, 1, 0)
            name = pic.split('.')[0] + '_aspn_noise.jpg'
            save_cv2(img_aspn, name, folder, target_path)

            img_aspn_flip = Image.fromarray(np.uint8(img_aspn))
            img_aspn_flip = img_aspn_flip.transpose(Image.FLIP_LEFT_RIGHT)
            name = pic.split('.')[0] + '_aspn_noise_flip.jpg'
            save(img_aspn_flip, name, folder, target_path)

            # 高斯滤波
            img_bl = filters.gaussian_filter(img, 0.5)
            img_bl = Image.fromarray(np.uint8(img_bl))
            name = pic.split('.')[0] + '_g_blur.jpg'
            save(img_bl, name, folder, target_path)

            img_bl_flip = img_bl.transpose(Image.FLIP_LEFT_RIGHT)
            name = pic.split('.')[0] + '_g_blur_flip.jpg'
            save(img_bl_flip, name, folder, target_path)
        i += 1
        print(str(i) + '@ ' + '扩充鞋印文件夹：' + folder + '完成')


# 高斯噪声
def AWN_image(image, NS):
    G_Noiseimg = image
    G_NoiseNum = int(NS*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(20,40)
        temp_y = np.random.randint(20,40)
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg


# 椒盐噪声
def ASPN_image(image, NS):
    img_ = image.transpose(2, 1, 0)
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[NS, (1 - NS) / 2., (1 - NS) / 2.])
    mask = np.repeat(mask, c, axis=0)
    img_[mask == 1] = 255
    img_[mask == 2] = 0
    return img_


# 保存cv2
def save_cv2(img, img_name, save_path, target_path):
    path = os.path.join(target_path, save_path)
    if not os.path.exists(path):
        os.makedirs(path)
    temp = path + '\\' + img_name
    if os.path.exists(temp):
        os.remove(temp)
    cv2.imwrite(temp, img)


# 保存图片
def save(img, img_name, save_path, target_path):
    path = os.path.join(target_path, save_path)
    if not os.path.exists(path):
        os.makedirs(path)
    temp = path + '\\' + img_name
    if os.path.exists(temp):
        os.remove(temp)
    img.save(temp)


if __name__ == '__main__':
    source_path = 'C:\\Users\\Neo\\Desktop\\source_bottom_300'
    target_path = 'C:\\Users\\Neo\\Desktop\\source_bottom_300_train'
    increase_data(source_path, target_path)