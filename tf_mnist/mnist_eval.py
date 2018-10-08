import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder('float', shape=[None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder('float', shape=[None, mnist_inference.OUTPUT_NODE], name='y-input')

        validation_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        y = mnist_inference.inference(x, None)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        acc = tf.reduce_mean((tf.cast(correct_pred, 'float')))

        variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_store = variable_average.variables_to_restore()
        saver = tf.train.Saver()

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    acc_score = sess.run(acc, feed_dict=validation_feed)

                    print('after %s training steps, validation accuracy = %g ' % (global_step, acc_score))

                else:
                    print('no checkpoint find')
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('..//MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
