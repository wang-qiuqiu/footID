FootID
===
Implementation of FootID using tensorflow.

# 简要说明

使用了鞋印数据集，在 8:1 的切分下达到了 99% 的识别精度。

# 环境配置
python3: `numpy, cv2, tensorflow`

# 运行步骤

`./prework.py` 定义数据集的根目录，8:1的比例进行训练集和测试集的切分，并保存data_label_npy文件夹为类别和标签的索引。以及foot_data_train文件夹训练集和测试集的data和label

`./footid1.py` 训练特征提取网络。

`./generate_feature.py` 使用footid1网络对训练集和测试集所有图片提取160维特征，分别保存在train_data_feature和test_data_feature文件夹中

`./featureID_dataGenerator.py` 按8:1的比例生成featureID的训练集和测试集，保存在feature_train文件夹中

`tensorboard --logdir=log(&feature_log)` 查看图模型及训练统计数据。

`./predict_foot.py` 对测试集数据，利用训练好的网络提取特征，

一些其它的超参数设置如下：

|梯度下降方法|初始学习率|激活函数|batch size|预测向量距离|
|----------|--------|-------|----------|----------|
|AdamOptimizer| 1e-4| sigmoid|      128|    cosine|

# 准确率

|迭代轮数|准确率|
|-----|------|
|10000|94.77%|
|20000|95.92%|
|30000|98.97%|

