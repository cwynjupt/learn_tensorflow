import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy.random import RandomState


def get_weight(shape, lamb):
    weight = tf.Variable(tf.random_normal(shape), dtype='float')
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(lamb)(weight))
    return weight


x = tf.placeholder('float', shape=[None, 2])
y_ = tf.placeholder('float', shape=[None, 1])

rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

layer_dem = [2, 10, 10, 1]
n_layer = len(layer_dem)

cur_layer = x
input_layer = layer_dem[0]

for i in range(1, n_layer):
    output_layer = layer_dem[i]
    weight = get_weight([input_layer, output_layer], 0.01)
    bias = tf.Variable(tf.constant(0.1, shape=[output_layer]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    input_layer = layer_dem[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

tf.add_to_collection('loss', mse_loss)

loss = tf.add_n(tf.get_collection('loss'))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

batch_size = 8
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for j in range(5000):
        start = (j * batch_size) % data_size
        end = min(start + batch_size, data_size)
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})

        if j % 100 == 0:
            now_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('after %d step, the loss is %g: ' % (j, now_loss))
