import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
import numpy as np
import matplotlib.pyplot as plt

class LSTMRNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])

        self.in_w1 = tf.Variable(tf.truncated_normal(shape=[28,128],stddev=0.01))
        self.in_b1 = tf.Variable(tf.zeros([128]))

        self.out_w2 = tf.Variable(tf.truncated_normal(shape=[128,10],stddev=0.01))
        self.out_b2 = tf.Variable(tf.zeros([10]))

    def forward(self):
        y = tf.reshape(self.x,shape=[-1,28])  #[100*28,28]
        y = tf.nn.relu(tf.matmul(y,self.in_w1) + self.in_b1)
        y = tf.reshape(y,shape=[-1,28,128])
        cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        init_state = cell.zero_state(100,dtype=tf.float32)
        outputs,final_state = tf.nn.dynamic_rnn(cell,y,initial_state=init_state,time_major=False)
        y = tf.transpose(outputs,[1,0,2])[-1]
        self.output = tf.nn.softmax(tf.matmul(y,self.out_w2) + self.out_b2)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def accuracys(self):
        bool = tf.equal(tf.argmax(self.output),tf.argmax(self.y))
        self.accuracy = tf.reduce_mean(tf.cast(bool,dtype=tf.float32))

if __name__ == '__main__':
    net = LSTMRNNNet()
    net.forward()
    net.backward()
    net.accuracys()
    init = tf.global_variables_initializer()

    plt.ion()
    a = []
    b = []

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000000):
            xs,ys = mnist.train.next_batch(100)
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs,net.y:ys})
            if i % 10 == 0:
                test_xs,test_ys = mnist.test.next_batch(100)
                output = sess.run(net.output,feed_dict={net.x:test_xs})

                print("loss:",loss)

                a.append(i)
                b.append(loss)
                plt.clf()
                plt.plot(a,b)
                
                plt.pause(0.1)



