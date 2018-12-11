import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1]) #nhwc
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        #定义卷积核的权重 偏值 hwc加神经元个数
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3,3,1,16]))
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3,3,16,32]))
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self):
        #卷积层和池化层
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="SAME") + self.conv1_b)  #第一次卷积之后 图像大小28*28
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  #第一次池化之后14*14

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding="SAME") + self.conv2_b) #第二次卷积  14*14
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  #第二次池化 7*7
        #计算图像大小 “（（n+2p-f ）/s +1）*(（n+2p-f ）/s +1)”结果向下取整  p = f-1/2  转换成NV
        self.flat = tf.reshape(self.pool2,shape=[-1,7*7*32])  #7*7是图像大小 32是通道
        #全连接
        self.w1 = tf.Variable(tf.truncated_normal(shape=[7*7*32,128],stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.w2 = tf.Variable(tf.truncated_normal(shape=[128,10],stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([10]))

        self.y1 = tf.nn.relu(tf.matmul(self.flat,self.w1)+self.b1)
        self.output = tf.nn.softmax(tf.matmul(self.y1,self.w2)+self.b2)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000000):
            xs,ys = mnist.train.next_batch(100)
            xss = tf.reshape(xs,[100,28,28,1])
            xsss=sess.run(xss)
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xsss,net.y:ys})
            print(loss)
            print(net.pool2.shape)

            # if i % 100 ==0:
            #     test_xs,test_ys = mnist.test.next_batch(100)
            #     test_xss = test_xs.reshape([100,28,28,1])
            #     output = sess.run(net.y,feed_dict={net.x:test_xss,net.y:test_ys})
            #     test_yss = np.argmax(test_ys)
            #     outputs = np.argmax(output)
            #     accuracy = np.mean(np.array(test_yss==outputs,dtype=np.float32))
            #     print("loss",loss)
            #     print("jingdu:",accuracy)
