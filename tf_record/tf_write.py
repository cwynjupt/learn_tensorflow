import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets('..//MNIST_data', dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.trian.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = 'result/tfrecord_output'

writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Feature(
        feature={
            'pixels': _int64_features(pixels),
            'label': _int64_features(labels),
            'image_raw': _bytes_feature(image_raw)
        }
    ))
    writer.write(example.SerializeToString())
writer.close()