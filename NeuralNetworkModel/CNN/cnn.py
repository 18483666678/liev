import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

save_path = r"D:\PycharmProjects\cnn2seq\CNN\save_ckpt"


class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        self.conv1_w = tf.Variable(tf.truncated_normal([3,3,1,16]))
        self.conv1_b = tf.Variable(tf.zeros([16]))
        self.conv2_w = tf.Variable(tf.truncated_normal([3,3,16,32]))
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="SAME") + self.conv1_b)
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding="SAME") + self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        self.flat = tf.reshape(self.pool2,[-1,7*7*32])
        self.in_w1 = tf.Variable(tf.truncated_normal([7*7*32,128],stddev=0.1))
        self.in_b1 = tf.Variable(tf.zeros([128]))
        self.out_w2 = tf.Variable(tf.truncated_normal([128,10],stddev=0.1))
        self.out_b2 = tf.Variable(tf.zeros([10]))
        self.out_y = tf.nn.relu(tf.matmul(self.flat,self.in_w1) + self.in_b1)
        self.output = tf.nn.softmax(tf.matmul(self.out_y,self.out_w2) + self.out_b2)

    def backward(self):
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        self.loss = tf.reduce_mean(tf.square(self.output - self.y))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

    def accuracys(self):
        self.bool = tf.equal(tf.argmax(self.y,axis=1),tf.argmax(self.output,axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.bool,dtype=tf.float32))

if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.backward()
    net.accuracys()
    init = tf.global_variables_initializer()
    save = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        save.restore(sess,save_path)
        for i in range(1000000):
            xs,ys = mnist.train.next_batch(100)
            xs_x = np.reshape(xs,[100,28,28,1])
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs_x,net.y:ys})
            if i % 100 ==0:
                xss,yss = mnist.test.next_batch(100)
                xss_s = np.reshape(xss,[100,28,28,1])
                output,acc = sess.run([net.output,net.accuracy],feed_dict={net.x:xss_s,net.y:yss})
                print("loss:",loss)
                print("acc:",acc)
                for j in range(10):
                    out = np.argmax(output[j])
                    label = np.argmax(yss[j])
                    print("output:",out,"label:",label)
                    print(np.mean(np.array(out==label,dtype=np.float32)))


                    # save.save(sess,save_path)
