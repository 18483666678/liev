import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[100,28,28,1]) #nhwc
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        #定义卷积核的权重 偏值 hwc加神经元个数
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3,3,1,16]))
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3,3,16,32]))
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self):
        #卷积层和池化层
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="SAME") + self.conv1_b)  #第一次卷积之后 图像大小28*28
        # self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  #第一次池化之后14*14

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1,self.conv2_w,strides=[1,1,1,1],padding="SAME") + self.conv2_b) #第二次卷积  14*14
        # self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  #第二次池化 7*7

        self.y1 = tf.layers.conv2d_transpose(self.conv2, 16, (3, 3), (1, 1), padding="SAME")
        # self.y2 = tf.layers.conv2d_trans+pose(self.pool1, 16, (2, 2), (2, 2), padding="SAME")
        self.y3 = tf.layers.conv2d_transpose(self.conv1,1, (3, 3), (1, 1), padding="SAME")
        # self.y4 = tf.layers.conv2d_transpose(self.pool2, 32, (2, 2), (2, 2), padding="SAME")




        #计算图像大小 “（（n+2p-f ）/s +1）*(（n+2p-f ）/s +1)”结果向下取整  p = f-1/2  转换成NV
        # self.flat = tf.reshape(self.pool2,shape=[-1,7*7*32])  #7*7是图像大小 32是通道
        #全连接
        # self.w1 = tf.Variable(tf.truncated_normal(shape=[7*7*32,128],stddev=0.1))
        # self.b1 = tf.Variable(tf.zeros([128]))
        #
        # self.w2 = tf.Variable(tf.truncated_normal(shape=[128,10],stddev=0.1))
        # self.b2 = tf.Variable(tf.zeros([10]))
        #
        # self.y1 = tf.nn.relu(tf.matmul(self.flat,self.w1)+self.b1)
        # self.output = tf.nn.softmax(tf.matmul(self.y1,self.w2)+self.b2)



if __name__ == '__main__':
    net = CNNNet()
    net.forward()


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(net.conv1.shape)
        print(net.conv2.shape)

        print(net.y1.shape)
        print(net.y3.shape)










