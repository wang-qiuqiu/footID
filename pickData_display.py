import os
import shutil


# 选择集外图片
def pick_outlier():
    count = 0
    std_path = 'C:\\Users\\Neo\\Desktop\\source_test'
    source_path = 'C:\\Users\\Neo\\Desktop\\source_display'
    target_path = 'C:\\Users\\Neo\\Desktop\\display_outlier'

    deleteFolder(target_path)

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
        folder_path = os.path.join(pick_path, folder)
        count = 0
        for pic in os.listdir(folder_path):
            if count == 2:
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


if __name__ == '__main__':
    main()