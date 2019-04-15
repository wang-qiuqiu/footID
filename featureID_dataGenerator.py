import numpy as np
import math
import os

target_classes = 2


def load_data(positive_data, negative_data):
    positive_data = np.array(positive_data)
    negative_data = np.array(negative_data)

    train_data = []
    train_label = []

    for i in range(positive_data.shape[0]):
        train_data.append(i)
        train_label.append(0)
    for i in range(negative_data.shape[0]):
        train_data.append(i+positive_data.shape[0])
        train_label.append(1)
    train_data = np.array(train_data)

    temp = np.array([train_data, train_label])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_data_list = temp[:, 0]
    all_label_list = list(temp[:, 1])
    all_label_list = [int(float(i)) for i in all_label_list]

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    ratio = 0.2
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))
    n_train = n_sample - n_val
    tra_data = all_data_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_data = all_data_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    val_label = val_labels

    val_data_list = []
    val_label_batch = np.zeros((len(val_label), target_classes))

    val_batch_index = 0
    for index in val_data:
        val_data_list.append(index)
        val_label_batch[val_batch_index][val_label[val_batch_index]] = 1
        val_batch_index += 1

    val_data_batch = np.array(val_data_list)

    save_path = 'feature_train'
    if os.path.exists(save_path):
        for file in os.listdir(save_path):
            temp_path = os.path.join(save_path, file)
            os.remove(temp_path)
        os.rmdir(save_path)
    os.makedirs(save_path)

    np.save(save_path + '/TrainX', tra_data)
    np.save(save_path + '/TrainY', tra_labels)
    np.save(save_path + '/TestX', val_data_batch)
    np.save(save_path + '/TestY', val_label_batch)


if __name__ == '__main__':
    train_feature = np.load('train_data_feature\\data_feature.npy')
    test_feature = np.load('test_data_feature\\data_feature.npy')

    train_feature = np.reshape(train_feature, (train_feature.shape[0], train_feature.shape[2], train_feature.shape[1]))
    test_feature = np.reshape(test_feature, (test_feature.shape[0], test_feature.shape[2], test_feature.shape[1]))

    load_data(train_feature, test_feature)