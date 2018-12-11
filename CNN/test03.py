import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32,[None,28,28,1])#注意CNN输入的数据形状NHWC
        self.y = tf.placeholder(tf.float32,[None,10])

        self.conv1_w = tf.Variable(tf.truncated_normal([3,3,1,16]))#3*3的卷积核，1个通道，16个特征（超参数）
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.truncated_normal([3,3,16,32]))
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="SAME")+self.conv1_b)
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding="SAME")+self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#7*7

        self.flat = tf.reshape(self.pool2,[-1,7*7*32])#-1表示剩下的所有，其他部分是图片的大小，和通道（重新整理图片）

        self.w1 = tf.Variable(tf.truncated_normal([7*7*32,128],stddev=tf.sqrt(1/64)))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.w2 = tf.Variable(tf.truncated_normal([128,10]))
        self.b2 = tf.Variable(tf.zeros([10]))

        self.y1 = tf.nn.relu(tf.matmul(self.flat,self.w1)+self.b1)
        self.out_put = tf.nn.softmax(tf.matmul(self.y1,self.w2)+self.b2)

    def backward(self):
        self.loss = tf.reduce_mean((self.out_put-self.y)**2)
        self.opt = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100000):
            xs,ys = mnist.train.next_batch(100)
            batch_xs = xs.reshape([100,28,28,1])#NHWC
            _loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:batch_xs,net.y:ys})
            print(_loss)
