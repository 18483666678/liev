import tensorflow as tf
import numpy as np
import os
import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import PIL.Image as pimg
import PIL.ImageDraw as pdraw
import PIL.ImageFont as Font

image_path = r"D:\PycharmProjects\cnn2seq\code"
font_path = r"D:\PycharmProjects\cnn2seq\arial.ttf"
save_path = r"D:\PycharmProjects\cnn2seq\save_cnn_ckpt/pt"
batch_size = 100


class EncoderNet:
    def __init__(self):
        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 3, 16],stddev=0.1))
        self.conv1_b = tf.Variable(tf.zeros([16]))
        self.conv2_w = tf.Variable(tf.truncated_normal([3, 3, 16, 32],stddev=0.1))
        self.conv2_b = tf.Variable(tf.zeros([32]))
        self.conv3_w = tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.1))
        self.conv3_b = tf.Variable(tf.zeros([64]))

    def forward(self, x):
        conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1_w, strides=[1, 1, 1, 1],padding="SAME") + self.conv1_b)  # 60*120  分别计算的60和120的padding
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 29*60  *32
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, self.conv2_w, strides=[1, 1, 1, 1], padding="SAME") + self.conv2_b)  # 29*60
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 15*30
        # conv3 = tf.nn.relu(tf.nn.conv2d(pool2,self.conv3_w,strides=[1,1,1,1],padding="SAME")) #15*30
        # pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #8*15

        # NV转换NSV
        flat = tf.reshape(pool2, shape=[-1, 15 * 30 * 32])
        # 全连接
        self.w1 = tf.Variable(tf.truncated_normal(shape=[15 * 30 * 32, 256]))
        self.b1 = tf.Variable(tf.zeros([256]))
        output = tf.matmul(flat, self.w1) + self.b1
        return output


class DecoderNet:
    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal(shape=[256, 10]))
        self.b1 = tf.Variable(tf.zeros([10]))

    def forward(self, x):
        y = tf.expand_dims(x, axis=1)
        y = tf.tile(y, [1, 4, 1])

        cell = tf.nn.rnn_cell.BasicLSTMCell(256)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell, y, initial_state=init_state, time_major=False)

        # y = tf.reshape(decoder_outputs, shape=[batch_size * 4, 128])
        y = tf.reshape(decoder_outputs,(-1,256))
        y = tf.nn.softmax(tf.matmul(y, self.w1) + self.b1)
        y = tf.reshape(y, shape=[-1, 4, 10])
        return y


class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 60, 120, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 4, 10])

        self.encoderNet = EncoderNet()
        self.decoderNet = DecoderNet()

    def forward(self):
        y = self.encoderNet.forward(self.x)
        self.output = self.decoderNet.forward(y)

    def backward(self):
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.output))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label=self.y,logits=self.output))
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.output),reduction_indices=[1]))
        self.loss = tf.reduce_mean((self.output - self.y) ** 2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)


class Sampling:
    def __init__(self):
        self.image_dataset = []
        for filename in os.listdir(image_path):
            x = imgplt.imread(os.path.join(image_path, filename)) / 255 - 0.5
            ys = filename.split(".")
            y = self.__one_hot(ys[0])
            self.image_dataset.append([x, y])

    def __one_hot(self, x):
        z = np.zeros(shape=(4, 10))
        for i in range(4):
            index = int(x[i])
            z[i][index] += 1
        return z

    def image_get_batch(self, size):
        xs = []
        ys = []
        for _ in range(size):
            index = np.random.randint(0, len(self.image_dataset))
            xs.append(self.image_dataset[index][0])
            ys.append(self.image_dataset[index][1])
        return xs, ys


if __name__ == '__main__':
    sample = Sampling()
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()

    plt.ion()
    a = []
    b = []

    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess,save_path)
        for i in range(1000000):
            xs, ys = sample.image_get_batch(batch_size)
            loss, _ = sess.run([net.loss, net.opt], feed_dict={net.x: xs, net.y: ys})
            if i % 100 == 0:
                xss, yss = sample.image_get_batch(batch_size)
                output = sess.run(net.output, feed_dict={net.x: xss, net.y: yss})
                print(np.shape(output))
                print("loss：", loss)
                for j in range(10):
                    out = np.argmax(output[j], axis=1)
                    lab = np.argmax(yss[j], axis=1)
                    print("output:", out)
                    print("label:", lab)
                    print("accuracy:", np.mean(np.array(out == lab, dtype=np.float32)))

                    # 标签和输出图片对比
                    # img = (xss[0]+0.5) * 255
                    # image = pimg.fromarray(np.uint8(img))
                    # imgdraw = pdraw.ImageDraw(image)
                    # font = Font.truetype(font_path, size=20)
                    # imgdraw.text(xy=(0, 0), text=str(out), fill="red", font=font)
                    # image.show()

                    # 损失图
                    # a.append(i)
                    # b.append(loss)
                    # plt.clf()
                    # plt.plot(a, b)
                    # plt.pause(0.01)
                    save.save(sess,save_path)
