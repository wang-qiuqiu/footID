import os
import shutil


# 选择集外图片
def pick_outlier():
    count = 0
    std_path = 'C:\\Users\\Neo\\Desktop\\source_test'
    source_path = 'C:\\Users\\Neo\\Desktop\\source_display'
    target_path = 'C:\\Users\\Neo\\Desktop\\display_outlier'

    if os.path.exists(target_path):
        if len(os.listdir(target_path)) > 0:
            for file in target_path:
                os.remove(os.path.join(target_path, file))
            os.rmdir(target_path)
        else:
            os.rmdir(target_path)
    os.makedirs(target_path)

    std_folder = []
    for folder in os.listdir(std_path):
        std_folder.append(folder)
    for folder in os.listdir(source_path):
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


def pick_inner():
    return


def main():
    pick_outlier()


if __name__ == '__main__':
    main()