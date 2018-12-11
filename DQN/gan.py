import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class DNet:
    def __init__(self):
        with tf.variable_scope("D_PARAM"):
            self.in_w = tf.Variable(tf.truncated_normal(shape=[784, 256], stddev=0.1))
            self.in_b = tf.Variable(tf.zeros([256]))
            self.out_w = tf.Variable(tf.truncated_normal(shape=[256, 1], stddev=0.1))
            self.out_b = tf.Variable(tf.zeros([1]))

    def forward(self, x):
        y = tf.nn.leaky_relu(tf.matmul(x, self.in_w) + self.in_b)
        y = tf.matmul(y, self.out_w) + self.out_b
        return y

    def getParams(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="D_PARAM")


class GNet:
    def __init__(self):
        with tf.variable_scope("G_PARAM"):
            self.in_w = tf.Variable(tf.truncated_normal(shape=[128, 7 * 7 * 32], stddev=0.1))
            self.in_b = tf.Variable(tf.zeros([7 * 7 * 32]))

            self.conv1_w = tf.Variable(tf.random_normal([2, 2, 64, 32], dtype=tf.float32, stddev=0.1))
            self.conv1_b = tf.Variable(tf.zeros([64]))
            self.conv2_w = tf.Variable(tf.random_normal([2, 2, 1, 64], dtype=tf.float32, stddev=0.1))
            self.conv2_b = tf.Variable(tf.zeros([1]))

            self.out_w = tf.Variable(tf.truncated_normal(shape=[14 * 14, 784]))
            self.out_b = tf.Variable(tf.zeros([784]))

    def forward(self, x):
        y = tf.nn.leaky_relu(tf.matmul(x, self.in_w) + self.in_b)
        y = tf.reshape(y, [-1, 7, 7, 32])
        y = tf.nn.leaky_relu(
            tf.nn.conv2d_transpose(y, self.conv1_w, output_shape=[100, 14, 14, 64], strides=[1, 2, 2, 1],
                                   padding="SAME") + self.conv1_b)
        y = tf.nn.leaky_relu(
            tf.nn.conv2d_transpose(y, self.conv2_w, output_shape=[100, 14, 14, 1], strides=[1, 1, 1, 1],
                                   padding="SAME") + self.conv2_b)
        y = tf.reshape(y, [-1, 14 * 14])
        y = tf.matmul(y, self.out_w) + self.out_b
        return y

    def getParams(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, scope="G_PARAM")


class Net:
    def __init__(self):
        self.r_x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.g_x = tf.placeholder(dtype=tf.float32, shape=[None, 128])

        self.dnet = DNet()
        self.gnet = GNet()

    def forward(self):
        self.out_d_y = self.dnet.forward(self.r_x)
        self.out_g_y = self.gnet.forward(self.g_x)
        self.out_d_g_y = self.dnet.forward(self.out_g_y)

    def backward(self):
        self.d_r_loss = tf.reduce_mean(tf.log(self.out_d_y))
        self.d_g_loss = tf.reduce_mean(tf.log(1 - self.out_d_g_y))
        self.d_loss = self.d_r_loss + self.d_g_loss
        self.d_opt = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.dnet.getParams())

        self.g_loss = tf.reduce_mean(-tf.log(self.out_g_y))
        self.g_opt = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.gnet.getParams())


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(1000000):
            r_xs, _ = mnist.train.next_batch(100)
            g_xs = np.random.normal(-1, 1, size=[100, 128])
            _d_loss, _g_loss, _ = sess.run([net.d_loss, net.g_loss, net.d_opt],
                                           feed_dict={net.r_x: r_xs, net.g_x: g_xs})

            g_xs = np.random.normal(-1, 1, size=[100, 128])
            sess.run(net.g_opt, feed_dict={net.g_x: g_xs})

            plt.ion()
            if epoch % 100 == 0:
                print("d_loss:", _d_loss, "g_loss:", _g_loss)

                # test_g_xs = np.random.normal(-1, 1, size=[1, 128])
                # test_img_data = sess.run(net.out_g_y, feed_dict={net.g_x: test_g_xs})
                # test_img = np.reshape(test_img_data, (28, 28))
                # plt.imshow(test_img)
                # plt.pause(0.1)
