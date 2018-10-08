import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import lenet_inference
import numpy as np

BATCH_SIZE = 100
LEARNNING_RATE_BASE = 0.01
LEARNNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'mnist_lenet5_model.ckpt'


def train(mnist):
    x = tf.placeholder('float', shape=[BATCH_SIZE, lenet_inference.IMAGE_SIZE,
                                       lenet_inference.IMAGE_SIZE, lenet_inference.NUM_CHANNELS], name='input-x')
    y_ = tf.placeholder('float', shape=[None, lenet_inference.OUTPUT_NODE], name='input-y')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = lenet_inference.inference(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    learning_rate = tf.train.exponential_decay(LEARNNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshaped = np.reshape(xs, (BATCH_SIZE, lenet_inference.IMAGE_SIZE, lenet_inference.IMAGE_SIZE,
                                          lenet_inference.NUM_CHANNELS))
            _, train_loss, step = sess.run([train_op, loss, global_step], feed_dict={x: xs_reshaped, y_: ys})

            if i % 1000 == 0:
                print('after %d step, the training loss is %g' % (step, train_loss))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('..//MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
