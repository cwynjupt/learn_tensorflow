import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import lenet_inference
import lenet_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder('float', [lenet_train.BATCH_SIZE, lenet_inference.IMAGE_SIZE,
                                     lenet_inference.IMAGE_SIZE, lenet_inference.NUM_CHANNELS], name='input-x')
        y_ = tf.placeholder('float', [None, lenet_inference.OUTPUT_NODE], name='input-y')

        validate_feed_x = mnist.validation.images
        validate_feed_y = mnist.validation.labels
        validate_feed_x = np.reshape(validate_feed_x, [lenet_train.BATCH_SIZE, lenet_inference.IMAGE_SIZE,
                                                       lenet_inference.IMAGE_SIZE, lenet_inference.NUM_CHANNELS])
        validate_feed = {x: validate_feed_x, y_: validate_feed_y}

        y = lenet_inference(x, None, None)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, 'float'))

        saver = tf.train.Saver()

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(lenet_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    acc_score = sess.run(acc, feed_dict=validate_feed)

                    print('after %d step, the acc is %g' % (global_step, acc_score))
                else:
                    print('no valid chackpoint')
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('..//MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
