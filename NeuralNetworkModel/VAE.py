import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class EncoderNet:

    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal(shape=[784,100],stddev=0.1))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.logvar_w = tf.Variable(tf.truncated_normal(shape=[100,128],stddev=0.1))
        self.mean_w = tf.Variable(tf.truncated_normal(shape=[100,128],stddev=0.1))

    def forward(self,x):
        y = tf.nn.relu(tf.matmul(x,self.in_w) + self.in_b)

        #两个输出  没加激活函数是因为不求概率
        mean = tf.matmul(y,self.mean_w)
        logvar = tf.matmul(y,self.logvar_w)
        return mean,logvar

class DecoderNet:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal(shape=[128,100],stddev=0.1))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.out_w = tf.Variable(tf.truncated_normal(shape=[100,784],stddev=0.1))
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
        #编码器返回两个值 均值和log方差 方差不能为负，用log方差
        self.mean,self.logVar = self.encoderNet.forward(self.x)
        I = tf.random_normal([128])  #I表示标准正态分布
        self.var = tf.exp(self.logVar) #把log方差变成方差
        std = tf.sqrt(self.var)  #标准差
        _x = std * I + self.mean  #解码器输入
        self.output = self.decoderNet.forward(_x)
        #这个过程叫做重整化

    #创建一个decode函数专门用来生成
    def decode(self):
        I = tf.random_normal(shape=[1,128])  #传入批次和特征
        return self.decoderNet.forward(I)

    def backward(self):
        loss_1 = tf.reduce_mean((self.output - self.x) ** 2 )
        loss_2 = tf.reduce_mean(0.5 * (-self.logVar + self.mean **2 +self.var - 1))
        self.loss = loss_1 + loss_2
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

if __name__ == '__main__':

    net = Net()
    test_output = net.decode() #测试输出
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        plt.ion()
        for epoch in range(1000000):
            xs,_ = mnist.train.next_batch(100)

            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs})

            if epoch % 100 == 0:
                test_img_data = sess.run(test_output)
                test_img = np.reshape(test_img_data,[28,28])
                plt.imshow(test_img)
                plt.pause(0.1)
                print("loss:",loss)

