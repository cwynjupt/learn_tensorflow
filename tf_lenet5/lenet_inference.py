import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1    # 图片的深度
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512


def get_weight_variable(shape, regularizer):
    # 只有全连接层的权重要加入正则化项
    weight = tf.get_variable('weight', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer is not None:
        tf.add_to_collection('loss', regularizer(weight))

    return weight


def get_bias_variable(shape, init):
    return tf.get_variable('bias', shape=shape, initializer=tf.constant_initializer(init))


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = get_weight_variable([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], None)
        conv1_bias = get_bias_variable([CONV1_DEEP], 0.0)

        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv1, conv1_bias)
        relu1 = tf.nn.relu(bias)

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = get_weight_variable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
        conv2_bias = get_bias_variable([CONV2_DEEP], 0.0)

        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv2, conv2_bias)

        relu2 = tf.nn.relu(bias)

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 由于第四层的输出为7*7*64的矩阵，而全连接层的输入为向量，所以需要将这个矩阵拉直成一个向量
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]   # 这里pool_shape[0]为一个batch中的数据量

    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weight = get_weight_variable([nodes, FC_SIZE], regularizer)
        fc1_bias = get_bias_variable([FC_SIZE], 0.1)

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight) + fc1_bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weight = get_weight_variable([FC_SIZE, NUM_LABELS], regularizer)
        fc2_bias = get_bias_variable([NUM_LABELS], 0.1)

        fc2 = tf.matmul(fc1, fc2_weight) + fc2_bias

    return fc2
