import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class EncoderNet:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal([784,100],stddev=0.01))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.logvar = tf.Variable(tf.truncated_normal([100,128],stddev=0.01))
        self.mean = tf.Variable(tf.truncated_normal([100,128],stddev=0.01))

    def forward(self,x):
        y = tf.nn.relu(tf.matmul(x,self.in_w) + self.in_b)
        mean = tf.matmul(y,self.logvar)
        logvar = tf.matmul(y,self.logvar)
        return mean,logvar

class DecoderNet:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal([128,100],stddev=0.01))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.out_w = tf.Variable(tf.truncated_normal([100,784],stddev=0.01))

    def forward(self,x):
        y = tf.nn.relu(tf.matmul(x,self.in_w) + self.in_b)
        return tf.matmul(y,self.out_w)

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])

        self.encoderNet = EncoderNet()
        self.decoderNet = DecoderNet()

        self.forward()
        self.backward()

    def forward(self):
        self.mean,self.logVar = self.encoderNet.forward(self.x)
        I = tf.random_normal(shape=[128])
        self.var = tf.exp(self.logVar)
        std = tf.sqrt(self.var)
        _x = std * I + self.mean
        self.output = self.decoderNet.forward(_x)

    def decode(self):
        I = tf.random_normal([1,128])
        return self.decoderNet.forward(I)

    def backward(self):
        loss_1 = tf.reduce_mean((self.output - self.x) ** 2)
        loss_2 = tf.reduce_mean(0.5 * (-self.logVar + self.mean ** 2 + self.var -1))
        self.loss = loss_1 + loss_2

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

if __name__ == '__main__':
    net = Net()
    test_output = net.decode()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        plt.ion()
        for epoch in range(1000000):
            xs,_ = mnist.train.next_batch(100)
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs})

            if epoch % 10 == 0:
                test_img_data = sess.run(test_output)
                test_img = np.reshape(test_img_data,[28,28])
                plt.imshow(test_img)
                plt.pause(0.1)
                print("loss:",loss)
