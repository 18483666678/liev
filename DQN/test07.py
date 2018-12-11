import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        self.w1 = tf.Variable(tf.truncated_normal(shape=[784,512],stddev=tf.sqrt(1/512),dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros(512),dtype=tf.float32)
        self.w2 = tf.Variable(tf.truncated_normal(shape=[512, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros(256), dtype=tf.float32)
        self.w3 = tf.Variable(tf.truncated_normal(shape=[256, 128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.b3 = tf.Variable(tf.zeros(128), dtype=tf.float32)
        self.w4 = tf.Variable(tf.truncated_normal(shape=[128, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.b4 = tf.Variable(tf.zeros(64), dtype=tf.float32)
        self.wo = tf.Variable(tf.truncated_normal(shape=[64, 10], stddev=tf.sqrt(1 / 10), dtype=tf.float32))
        self.bo = tf.Variable(tf.zeros(10), dtype=tf.float32)

    def forward(self):
            self.y1 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.x,self.w1)+self.b1))
            self.y2 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.y1, self.w2) + self.b2))
            self.y3 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.y2, self.w3) + self.b3))
            self.y4 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.y3, self.w4) + self.b4))
            self.yo = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.y4, self.wo) + self.bo))
    def backward(self):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.yo))
            tf.summary.scalar("loss",self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    def summary(self,name,w):
        tf.summary.histogram(name+"_w",w)
        tf.summary.scalar(name+"_max",tf.reduce_max(w))
        tf.summary.scalar(name+"_min",tf.reduce_min(w))
        tf.summary.scalar(name+"_mean",tf.reduce_mean(w))



if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()



    init = tf.global_variables_initializer()

    merged = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter("./logs",sess.graph)
        for i in range(5000):
            xs,ys = mnist.train.next_batch(500)
            summery,error,_ = sess.run([merged,net.loss,net.optimizer],feed_dict={net.x:xs,net.y:ys})

            writer.add_summary(summery,i)
            # if i in range(200):
            #     xss,yss = mnist.train.next_batch(500)
            #     _error, _out = sess.run([net.loss, net.yo], feed_dict={net.x: xss, net.y: yss})
            #     print(_error)

                # label = np.argmax(yss[0])
                # output = np.argmax(_out[0])
                # print("label:",label,"==>output:",output)
