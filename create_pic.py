import os
import matplotlib.pyplot as plt


#   图2.3
def chap2_2_3(pic_dir):
    pic_count_all = []
    for folder in os.listdir(pic_dir):
        pic_count = 0
        temp_dir = os.path.join(pic_dir, folder)
        for pic in os.listdir(temp_dir):
            pic_count += 1
        pic_count_all.append(pic_count)
    list = sorted(pic_count_all)
    draw_hist(list)


def draw_hist(count_list):
    x_ = [i for i in range(len(count_list))]
    y_ = count_list
    plt.plot(x_, y_, linewidth=2)
    plt.title('Distribution of Shoeprint', fontsize=10)
    plt.xlabel('Classes', fontsize=7)
    plt.ylabel('Quantity', fontsize=7)
    plt.tick_params(axis='both', labelsize=7)
    plt.show()


def main():
    dir = 'C:\\Users\\Neo\Desktop\\train'
    list = chap2_2_3(dir)
    return


if __name__ == '__main__':
    main()