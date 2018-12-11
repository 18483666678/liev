import tensorflow as tf
import os
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np

iamge_path = r"G:\PycharmProjects\1028\tensorflowlianxi\code"
font_path = r"G:\PycharmProjects\1028\tensorflowlianxi\arial.ttf"
save_path = r"G:\PycharmProjects\1028\tensorflowlianxi\save_cnn_ckpt\pt"
batch_size = 10

class EncoderCNNNet:
    def __init__(self):
        self.conv1_w = tf.Variable(tf.truncated_normal([3,3,3,16]))
        self.conv1_b = tf.Variable(tf.zeros([16]))
        self.conv2_w = tf.Variable(tf.truncated_normal([3,3,16,32]))
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self,x):
        conv1 = tf.nn.relu(tf.nn.conv2d(x,self.conv1_w,strides=[1,1,1,1],padding="SAME") + self.conv1_b)
        pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1,self.conv2_w,strides=[1,1,1,1],padding="SAME"))
        pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
        flat = tf.reshape(pool2,[-1,15*30*32])
        self.w1 = tf.Variable(tf.truncated_normal([15*30*32,256]))
        self.b1 = tf.Variable(tf.zeros([256]))
        output = tf.matmul(flat,self.w1) + self.b1
        return output

class DecoderNet:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([256,10]))
        self.b1 = tf.Variable(tf.zeros([10]))

    def forward(self,x):
        y = tf.expand_dims(x,axis=1)
        y = tf.tile(y,[1,4,1])
        cell = tf.nn.rnn_cell.BasicLSTMCell(256)
        init_state =  cell.zero_state(batch_size,dtype=tf.float32)
        D_outputs,final_state = tf.nn.dynamic_rnn(cell,y,initial_state=init_state,time_major=False)
        y = tf.reshape(D_outputs,[batch_size*4,256])
        self.y1 = tf.matmul(y,self.w1) + self.b1
        self.y1 = tf.reshape(self.y1,[-1,4,10])
        y = tf.nn.softmax(self.y1)
        return y

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[batch_size,60,120,3])
        self.y = tf.placeholder(dtype=tf.float32,shape=[batch_size,4,10])

        self.encoderCNNNet = EncoderCNNNet()
        self.decoderNet = DecoderNet()

    def forward(self):
        y = self.encoderCNNNet.forward(self.x)
        self.output = self.decoderNet.forward(y)

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.decoderNet.y1,labels=self.y))
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

class Sampling:
    def __init__(self):
        self.image_dataset = []
        for filename in os.listdir(iamge_path):
            #批量归一化
            x = imgplt.imread(os.path.join(iamge_path,filename)) / 255 - 0.5
            ys = filename.split(".")
            y = self.__one_hot(ys[0])
            self.image_dataset.append([x,y])

    def __one_hot(self,x):
        z = np.zeros(shape=(4,10))
        for i in range(4):
            index = int(x[i])
            z[i][index] += 1
        return z
    def image_get_batch(self,size):
        xs = []
        ys = []
        for i in range(size):
            index = np.random.randint(0,len(self.image_dataset))
            xs.append(self.image_dataset[index][0])
            ys.append(self.image_dataset[index][1])
        return xs,ys

if __name__ == '__main__':
    sample = Sampling()
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess,save_path)
        for i in range(1000000):
            xs,ys = sample.image_get_batch(batch_size)
            loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:xs,net.y:ys})
            if i%10 ==0:
                xss,yss = sample.image_get_batch(batch_size)
                output = sess.run(net.output,feed_dict={net.x:xss,net.y:yss})
                print("loss:",loss)
                output = np.argmax(output,axis=1)
                label = np.argmax(yss,axis=1)
                print("accuracy:",np.mean(np.array(output==label,dtype=np.float32)))
                print("output:{0},label:{1}".format(output,yss))
                save.save(sess,save_path)