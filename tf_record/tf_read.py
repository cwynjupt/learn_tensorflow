import tensorflow as tf

reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer(['result/tfrecord_output'])

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'labels': tf.FixedLenFeature([], tf.int64)
    }
)

images = tf.decode_raw(features['image_raw'], tf.uint8)
pixels = tf.cast(features['pixels'], tf.int32)
labels = tf.cast(features['labels'], tf.int32)

sess = tf.Session()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    images, labels, pixels = sess.run([images, labels, pixels])

sess.close()