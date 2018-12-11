import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MSINT_data/",one_hot=True)

class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32,[None,28,28,1]) #shape=[None,28,28,1]四维NHWC 28,28是图片尺寸，1是通道数
        self.y = tf.placeholder(tf.float32,[None,10])

        self.conv1_w = tf.Variable(tf.truncated_normal([3,3,1,16]))  #卷积核[3,3,1,16] 3*3的卷积核，1个通道，16个特征（超参数一般16,24,32）
        self.conv1_b = tf.Variable(tf.zeros([16])) #多少个特征就多少个卷积核

        #第二层的卷积核
        self.conv2_w = tf.Variable(tf.truncated_normal([3,3,16,32])) #16是通道，32个特征
        self.conv2_b = tf.Variable(tf.zeros([32]))
    def forward(self):
        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding="SAME") + self.conv1_b)
        self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #ksize窗口大小，strides步长

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pool1,self.conv2_w,strides=[1,1,1,1],padding="SAME") + self.conv2_b)
        self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")  #池化层  最后一次池化完长宽为7*7

        #NHWC转化成NV结构
        self.flat = tf.reshape(self.pool2,[-1,7*7*32]) #[-1,7*7*32] NV结构 N批次用-1自动补全，V7*7*32 7*7是卷积步长的大小，32个通道
        #-1表示剩下的所有，其他部分是图片的大小，和通道的个数（重新整理图片）

        # 全连接层
        self.w1 = tf.Variable(tf.truncated_normal([7*7*32,128],stddev=tf.sqrt(1/64)))
        self.b1 = tf.Variable(tf.zeros([128]))

        self.w2 = tf.Variable(tf.truncated_normal([128,10],stddev=tf.sqrt(1/5)))
        self.b2 = tf.Variable(tf.zeros([10]))

        self.y1 = tf.nn.relu(tf.matmul(self.flat,self.w1) + self.b1)
        self.out_put = tf.nn.softmax(tf.matmul(self.y1,self.w2) + self.b2)

    def backwawrd(self):
        self.loss = tf.reduce_mean((self.out_put-self.y)**2)
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

if __name_ == '__main__':
    net = CNNNet()
    net.forward()
    net.backwawrd()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100000):
            xs,ys = mnist.train.next_batch(100)
            batvh_xs = xs.reshape([100,28,28,1])  #NV转化NHWC
            _loss,_ = sess.run([net.loss,net.opt],feed_dict={net.x:batvh_xs,net.y:ys})
            if i % 100 == 0:
                print(_loss)