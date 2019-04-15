import numpy as np
import tensorflow as tf
import cv2

# 获取所有的训练图片
TrainX = np.load('display_train\\TrainX.npy')
TrainY = np.load('display_train\\TrainY.npy')
TestX = np.load('display_train\\TestX.npy')
TestY = np.load('display_train\\TestY.npy')

class_num = 300
batch_size = 256


# 权重初始化
def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))


# 偏置初始化
def bias_variable(shape):
    with tf.name_scope('biases'):
        #权重都初始化为0.1
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))


# 定义卷积操作
def Wx_plus_b(weights, x, biases):
    with tf.name_scope('Wx_plus_b'):
        return tf.matmul(x, weights) + biases


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != None:
            activations = act(preactivate, name='activation')
            return activations
        else:
            return preactivate


# 定义卷积池化层
def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.sigmoid, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        sigmoid = act(h, name='sigmoid')
        if only_conv == True:
            return sigmoid
        pool = tf.nn.avg_pool(sigmoid, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='avg_pool')
        return pool


# 计算准确率
def accuracy(y_estimate, y_real):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            # 对这里的y和y_有点疑问？
            # tf.argmax()返回的是向量中最大值的索引
            # #tf.equal()两个向量，对应位置相等的话返回true否则false
            correct_prediction = tf.equal(tf.argmax(y_estimate, 1), tf.argmax(y_real, 1))
        with tf.name_scope('accuracy'):
            # tf.cast()将bool型的转换成float类型
            # tf.reduce_mean()返回均值
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy


def train_step(loss):
    with tf.name_scope('train'):
        # 学习率0.0001
        return tf.train.AdamOptimizer(1e-4).minimize(loss)


with tf.name_scope('input'):
    # h0为输入占位符，图片尺寸需要修改
    h0 = tf.placeholder(tf.float32, [None, 64, 32, 3], name='x')
    #y_为标签
    y_ = tf.placeholder(tf.float32, [None, class_num], name='y')

h1 = conv_pool_layer(h0, [5, 5, 3, 20], [20], 'Conv_layer_1')
h2 = conv_pool_layer(h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
h3 = conv_pool_layer(h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
h4 = conv_pool_layer(h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)

with tf.name_scope('FootID1'):
    h3r = tf.reshape(h3, [-1, 6 * 2 * 60])
    h4r = tf.reshape(h4, [-1, 5 * 1 * 80])

    W1 = weight_variable([6 * 2 * 60, 160])
    W2 = weight_variable([5 * 1 * 80, 160])

    b = bias_variable([160])
    h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
    h5 = tf.nn.sigmoid(h)

tf.add_to_collection('foot_feature', h5)

with tf.name_scope('loss'):
    y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('loss', loss)

accuracy = accuracy(y, y_)
train_step = train_step(loss)

merged = tf.summary.merge_all()
foot_saver = tf.train.Saver()

if __name__ == '__main__':

    trn_datas = TrainX
    trn_labels = TrainY

    logdir = 'display_log'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)
    
    sess2 = tf.Session()
    sess2.run(tf.initialize_all_variables())
    train_writer = tf.summary.FileWriter(logdir + '/train', sess2.graph)
    test_writer = tf.summary.FileWriter(logdir + '/test', sess2.graph)
    
    idx = 0
    k = 0
    for i in range(50001):
        ###################### 读取图片，生成batch
        batch_x = []
        batch_y = np.zeros((batch_size, class_num))
        if k == len(trn_labels)//batch_size:
            k = 0
        batch_x_ = trn_datas[batch_size * k:batch_size * (k + 1)]
        batch_y_ = np.array(trn_labels[batch_size * k:batch_size * (k + 1)])
        batch_row = 0
        for file_path in batch_x_:
            # 读入图片并转化为灰度图
            img = cv2.imread(file_path)
            img = np.array(img).astype(np.float32)
            batch_x.append(img)
            batch_y[batch_row][batch_y_[batch_row]] = 1
            batch_row += 1
        batch_x = np.array(batch_x)
        k += 1
        ################################
        # 添加准确率和loss的输出
        summary, _, trn_acc, trn_loss = sess2.run([merged, train_step, accuracy, loss], {h0: batch_x, y_: batch_y})
        # 添加在控制台上打印loss和acc
        print(str(i) + ": train --->" + "loss: " + str(trn_loss) + ", accuracy: " + str(trn_acc))
        train_writer.add_summary(summary, i)
    
        if i % 100 == 0 and i != 0:
            # 测试数据集太大导致OOM异常
            #######################
            a = TestX[0:2000, :, :, :]
            b = TestY[0:2000, :]
            #######################
            summary, test_acc = sess2.run([merged, accuracy], {h0: a, y_: b})
            print(str(i) + ": test --->" + "accuracy: " + str(test_acc))
            test_writer.add_summary(summary, i)
        if i % 10000 == 0 and i != 0:
            foot_saver.save(sess2, 'display_checkpoint/%05d.ckpt' % i)
