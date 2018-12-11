import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class EncoderNet:
    def __init__(self):
        self.in_w = tf.Variable(tf.truncated_normal(shape=[784,100],stddev=0.1))
        self.in_b = tf.Variable(tf.zeros([100]))

        self.mean_w = tf.Variable(tf.truncated_normal(shape=[100,128],stddev=0.1))
        self.logvar_w = tf.Variable(tf.truncated_normal(shape=[100,128],stddev=0.1))

    def forward(self,x):
        y = tf.nn.relu(tf.matmul(x,self.in_w) + self.in_b)
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

    def forward(self):
        self.mean,self.logVar = self.encoderNet.forward(self.x)
        I = tf.random_normal(shape=[128])
        self.var = tf.exp(self.logVar)  #把log方差变成方差
        std = tf.sqrt(self.var)  #标准差  方差开方之后是标准差
        _x = std * I + self.mean #标准正态分布变成另外一个正态分布 公式：std*I + mean 标准差乘以标准正态分布加上均值
        self.output = self.decoderNet.forward(_x)

    def backward(self):
        loss_1 = tf.reduce_mean((self.output - self.x) ** 2) #均方差
        loss_2 = tf.reduce_mean(0.5 * (-self.logVar + self.mean ** 2 + self.var -1)) #KL散度（相对熵）
        self.loss = loss_1 + loss_2
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    #训练完后 调用解码器（不用编码器），输入特征生成
    def decode(self):
        I = tf.random_normal(shape=[1,128])
        return self.decoderNet.forward(I)

if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    test_output = net.decode()  #测试输出
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        plt.ion()
        for epoch in range(1000000):
            xs,_ = mnist.train.next_batch(100)
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs})
            if epoch % 100 == 0:
                xss,_ = mnist.test.next_batch(100)
                test_img_data = sess.run(test_output)
                test_img = np.reshape(test_img_data,[28,28])
                plt.imshow(test_img)
                plt.pause(0.1)
                print("loss:",loss)