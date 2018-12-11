import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MSINT_data/",one_hot=True)

class CNNNet:
    def __init__(self):
        pass
    def forward(self):
        pass
    def backwawrd(self):
        pass

if __name__ == '__main__':
    net = CNNNet()
    net.forward()
    net.backwawrd()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)