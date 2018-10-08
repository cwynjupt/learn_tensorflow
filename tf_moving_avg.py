import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

v = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_average_op = ema.apply([v])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run([v, ema.average(v)]))

    sess.run(v.assign(5))
    print(sess.run(v))
    sess.run(maintain_average_op)
    print(sess.run([v, ema.average(v)]))

    sess.run(step.assign(10000))
    sess.run(v.assign(1))
    sess.run(maintain_average_op)
    print(sess.run([v, ema.average(v)]))