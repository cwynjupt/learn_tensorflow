import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER_NODE = [INPUT_NODE, 500, OUTPUT_NODE]

LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weight = tf.get_variable('weight', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('loss', regularizer(weight))

    return weight


def inference(input_tensor, regularizer):
    layer = []
    for i in range(len(LAYER_NODE) - 1):
        print(layer)
        shape_weight = [LAYER_NODE[i], LAYER_NODE[i+1]]
        shape_bias = [LAYER_NODE[i+1]]
        print(shape_weight, shape_bias)
        with tf.variable_scope('layer' + str(i + 1)):
            weight = get_weight_variable(shape_weight, regularizer)
            bias = tf.get_variable('bias', shape=shape_bias, initializer=tf.constant_initializer(0.0))
            if i == 0:
                layer.append(tf.nn.relu(tf.matmul(input_tensor, weight) + bias))
            elif i + 1 == len(LAYER_NODE):
                layer.append(tf.matmul(layer[-1], weight) + bias)
            else:
                layer.append(tf.nn.relu(tf.matmul(layer[-1], weight) + bias))
    print(layer, layer[-1])
    return layer[-1]


def inference_(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weight = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        bias = tf.get_variable('bias', shape=[LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + bias)

    with tf.variable_scope('layer2'):
        weight = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        bias = tf.get_variable('bias', shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weight) + bias
    print(layer1, layer2)
    return layer2
