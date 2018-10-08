import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')

    result = v1 + v2

    # 保存模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(result)
        saver.save(sess, 'model/save_test.ckpt')
        saver.export_meta_graph('model/save_test.ckpt.meta.json', as_text=True)


# 加载刚刚保存的模型，需要使用和保存模型中一样的代码来声明变量，但是不用初始化
def restore():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, 'model/save_test.ckpt')
        print('restore', sess.run(result))


# 加载刚刚保存的模型，变量命名不一样，需要重新命名
def restore_1():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v_1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v_2')
    result = v1 + v2
    with tf.Session() as sess:
        saver = tf.train.Saver({'v1': v1, 'v2': v2})
        saver.restore(sess, 'model/save_test.ckpt')
        print('restore', sess.run(result))


# 直接加载持久化的图
def restore_2():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/save_test.ckpt.meta')
        saver.restore(sess, 'model/save_test.ckpt')
        print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))


def checkpoint():
    reader = tf.train.NewCheckpointReader('model/save_test.ckpt')

    all_variables = reader.get_variable_to_shape_map()
    for variable_name in all_variables:
        print(variable_name, all_variables[variable_name])

    print('value of variable v1 is ', reader.get_tensor('v1'))


# save()
# restore()
# restore_1()
checkpoint()
