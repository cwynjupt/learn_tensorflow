from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)


import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder('float', [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

'''我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。'''
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

'''现在，我们已经设置好了我们的模型。在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：'''
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

'''
该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处
理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''
首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向
量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 
代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

'''这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print('the accuracy of tensorflow 1 is', sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels}))
