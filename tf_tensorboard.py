import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], name='a')
b = tf.Variable(tf.random_uniform([3]), name='b')
c = tf.add_n([a, b], name='c')

writer = tf.summary.FileWriter('log', tf.get_default_graph())
writer.close()
