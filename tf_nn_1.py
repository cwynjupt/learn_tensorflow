import tensorflow as tf
from numpy.random import RandomState
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

x = tf.placeholder('float', shape=(None, 2), name='x_input')
y_ = tf.placeholder('float', shape=(None, 1), name='y_input')

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

rdm = RandomState(1)
data_set_size = 128
X = rdm.rand(data_set_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
print(X)
print(Y)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print('ini_w1: ', sess.run(w1))
    print('ini_w2: ', sess.run(w2))
    step = 5000
    batch_size = 8
    for i in range(step):
        start = (i * batch_size) % data_set_size
        end = min(start + batch_size, data_set_size)
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print('after %d step,cross-entropy on all data set is %g' % (i, total_cross_entropy))

    print('final_w1: ', sess.run(w1))
    print('final_w2: ', sess.run(w2))
