import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
# LAYER_NODE = [10]

LAYER_NODE = [INPUT_NODE, 10, OUTPUT_NODE]


def inference_(input_tensor, avg_class, reuse=False):
    layer = []
    for i in range(len(LAYER_NODE) - 2):
        if avg_class is None:
            with tf.variable_scope('layer' + str(i), reuse=reuse):
                weight = tf.get_variable('weight', shape=[LAYER_NODE[i], LAYER_NODE[i + 1]],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
                bias = tf.get_variable('bias', shape=[LAYER_NODE[i + 1]], initializer=tf.constant_initializer(0.0))
                if i == 0:
                    layer.append(tf.nn.relu(tf.matmul(input_tensor, weight) + bias))
                elif i != len(LAYER_NODE) - 2:
                    layer.append(tf.nn.relu(tf.matmul(layer[-1], weight) + bias))
                else:
                    layer.append(tf.matmul(layer[-1], weight) + bias)
        
        else:
            with tf.variable_scope('layer' + str(i), reuse=reuse):
                weight = tf.get_variable('weight', shape=[LAYER_NODE[i], LAYER_NODE[i + 1]],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
                bias = tf.get_variable('bias', shape=[LAYER_NODE[i + 1]], initializer=tf.constant_initializer(0.0))
                if i == 0:
                    layer.append(tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight)) + avg_class.average(bias)))
                elif i != len(LAYER_NODE) - 2:
                    layer.append(tf.nn.relu(tf.matmul(layer[-1], avg_class.average(weight)) + avg_class.average(bias)))
                else:
                    layer.append(tf.matmul(layer[-1], avg_class.average(weight)) + avg_class.average(bias))

    return layer[-1]


