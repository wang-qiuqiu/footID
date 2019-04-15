import os
from PIL import Image


def crop_img_by_half_center(src_img_path):
    im = Image.open(src_img_path)
    x_size, y_size = im.size
    start_point_xy = x_size / 4
    end_point_xy = x_size / 4 + x_size / 2
    start_point_yx = y_size / 4
    end_point_yx = y_size / 4 + y_size / 2
    box = (start_point_xy, start_point_yx, end_point_xy, end_point_yx)
    new_im = im.crop(box)
    new_im = new_im.resize((32, 64))
    return new_im


def walk_through_the_folder_for_crop(source_folder, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(source_folder):
        print(source_folder + ' 源文件目录不存在')
        return
    for folder in os.listdir(source_folder):
        temp_folder = os.path.join(source_folder, folder)
        target_folder = os.path.join(save_path, folder)
        if os.path.exists(target_folder):
            for file in target_folder:
                os.remove(os.path.join(target_folder, file))
            os.rmdir(target_folder)
        os.makedirs(target_folder)
        for raw_img in os.listdir(temp_folder):
            raw_img_path = os.path.join(temp_folder, raw_img)

            # 取中间部分
            # new_img = crop_img_by_half_center(raw_img_path)

            # 取脚掌部分
            # new_img = crop2top(raw_img_path)

            # 取脚跟部分
            new_img = crop2bottom(raw_img_path)

            new_img.save(target_folder + '\\' + raw_img)
            print('生成新图片：' + raw_img_path + ' 完成')


# 整张图裁剪脚掌
def crop2top(src_img_path):
    im = Image.open(src_img_path)
    x_size, y_size = im.size
    start_point_xy = 0
    end_point_xy = x_size
    start_point_yx = 0
    end_point_yx = y_size / 2
    box = (start_point_xy, start_point_yx, end_point_xy, end_point_yx)
    new_im = im.crop(box)
    return new_im


# 裁剪得到脚跟
def crop2bottom(src_img_path):
    im = Image.open(src_img_path)
    x_size, y_size = im.size
    start_point_xy = 0
    end_point_xy = x_size
    start_point_yx = (2*y_size) / 3
    end_point_yx = y_size
    box = (start_point_xy, start_point_yx, end_point_xy, end_point_yx)
    new_im = im.crop(box)
    return new_im


if __name__ == '__main__':
    # # 得到脚掌部分
    # source_path = 'C:\\Users\\Neo\\Desktop\\source_test'
    # save_path = 'C:\\Users\\Neo\\Desktop\\source_top_300'
    # walk_through_the_folder_for_crop(source_path, save_path)

    # # 得到脚跟部分
    source_path = 'C:\\Users\\Neo\\Desktop\\source_test'
    save_path = 'C:\\Users\\Neo\\Desktop\\source_bottom_300_raw'
    walk_through_the_folder_for_crop(source_path, save_path)