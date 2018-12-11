import tensorflow as tf

class EncoderNet:

    def __init__(self):
        self.w = tf.Variable(tf.truncated_normal([784,100]))
        self.b = tf.Variable(tf.zeros([100]))

        self.mean_w = tf.Variable(tf.truncated_normal([100,128]))
        self.logVar_w = tf.Variable(tf.truncated_normal([100,128]))

    def forward(self,x):
        y = tf.nn.relu(tf.matmul(x,self.w) + self.b)
        self.mean = tf.matmul(y,self.mean_w)
        self.logVar = tf.matmul(y,self.logVar_w)

class DecoderNet:

    def __init__(self):
        pass

    def forward(self):
        pass

class Net:

    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,784])

        self.encoderNet = EncoderNet()
        self.decoderNet = DecoderNet()

    def forward(self):
        self.encoderNet.forward(self.x)

    def backward(self):
        pass

if __name__ == '__main__':
    net = Net()