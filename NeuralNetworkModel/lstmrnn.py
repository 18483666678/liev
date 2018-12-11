import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class LstmRNN:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,10])

        #NV变成N（100,28） V（28） 后V的第一层权重：28*128（要变成的权重）
        self.in_w1 = tf.Variable(tf.truncated_normal([28,128],stddev=0.01))
        self.in_b1 = tf.Variable(tf.zeros([128]))

        #输出层神经元权重
        self.out_w2 = tf.Variable(tf.truncated_normal([128,10],stddev=0.01))
        self.out_b2 = tf.Variable(tf.zeros([10]))

    def forward(self):
        #变形合并前的形状是[100,784]-->[100,28,28],
        # 合并后的形状[100*28,28]  //NV-->NSV-->N(NS)V
        y = tf.reshape(self.x,[-1,28])

        #[100*28,28]*[28,128]第一层计算后的形状[100*28,128] //   N(NS)V
        y = tf.nn.relu(tf.matmul(y,self.in_w1)+self.in_b1)

        #第一层计算后再变形的形状是从[100*28,128]-->[100,28,128]  //N(NS)V -->NSV
        y  = tf.reshape(y,[-1,28,128])

        #实例化单层网络（记忆细胞的神经元个数）超级参数
        cell = tf.nn.rnn_cell.BasicLSTMCell(128)

        #初始化每一批次记忆细胞的状态（超级参数）
        init_state = cell.zero_state(100,dtype=tf.float32)

        #outputs：LSTM展开后的每一个y的值，final_state:最后一个记忆细胞的值。outputs的值是NSV结构。传入的是记忆细胞的个数，最后输入的y值
        #双向rnn tf.nn.bidirectional_dynamic_rnn()
        outputs,final_state = tf.nn.dynamic_rnn(cell,y,initial_state=init_state,time_major=False)

        #矩阵形状转置，NSV转置成SNV，下标调换，取最后一组（s的最后一步）数据获取结果。就变成了NV
        y = tf.transpose(outputs,[1,0,2])[-1]  #[100,28,128]-->[28,100,128]
        # y = outputs[:,-1,:] #和上面一样的效果
        #输出最终形状为是从[100,28,128]-->（转置SNV）-->[28,100,128]-->(只取最后一张图片：步长从28变为1)-->[1*100,128]=[100,128]
        self.output = tf.nn.softmax(tf.matmul(y,self.out_w2)+self.out_b2)

    def backward(self):
        #交叉熵算损失
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.y))
        #均方差算损失
        self.loss = tf.reduce_mean(tf.square(self.output-self.y))
        #Adam优化器
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

if __name__ == '__main__':
    net = LstmRNN()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(5000):
            xs,ys = mnist.train.next_batch(100)
            loss,_ = sess.run([net.loss,net.optimizer],feed_dict={net.x:xs,net.y:ys})
            if i % 10 == 0:
                test_xs,test_ys = mnist.test.next_batch(100)
                test_output = sess.run(net.output,feed_dict={net.x:test_xs,net.y:test_ys})
                print("loss:",loss)
                print(type(test_output))

                #标签对比  精度
                y_output = np.argmax(test_output,1)
                y_label = np.argmax(test_ys,1)

                print("accuracy:",np.mean(np.array(y_output==y_label,dtype=np.float32)))
                print("label:",y_label[0],"out_put:",y_output[0])


