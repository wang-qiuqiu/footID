import shutil
import os

root = 'C:\\Users\\Neo\\Desktop\\source_test'
target = 'C:\\Users\\Neo\\Desktop\\display'


# 产生测试数据
def get_data(source, target):
    if os.path.exists(target):
        for file in os.listdir(target):
            file_path = os.path.join(target, file)
            os.remove(file_path)
        os.rmdir(target)
    os.makedirs(target)
    for folder in os.listdir(source):
        temp_folder = os.path.join(source, folder)
        for pic in os.listdir(temp_folder):
            temp_pic = os.path.join(temp_folder, pic)
            shutil.copy(target, temp_pic)


def main():
    get_data(root, target)


if __name__ == '__main__':
    main()