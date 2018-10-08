import tensorflow as tf
from tensorflow.python.framework import graph_util


def save():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[2]), name='v2')

    result = v1 + v2

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # sess.run(result)

        graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

        with tf.gfile.GFile('model\convert_variable_to_constant.pd', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


from tensorflow.python.platform import gfile
def restore():
    with tf.Session() as sess:
        model_filename = 'model\convert_variable_to_constant.pd'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))


restore()