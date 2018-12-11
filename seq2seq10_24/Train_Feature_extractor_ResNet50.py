import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pimg
import PIL.ImageDraw as pdraw
import PIL.ImageFont as fonts
from project_mtcnn.tools import until
from project_mtcnn.tools import c_loss
from project_mtcnn.tools import listes
from project_mtcnn.tools import one_hot
from project_mtcnn.sample import sampling_feature_train_record
from project_mtcnn.sample import sampling_feature_validation_record
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

train_filename = r'E:\pycharm_project\project_mtcnn\sample\feature_train.tf_record'  # 输出文件地址
validation_filename = r'E:\pycharm_project\project_mtcnn\sample\feature_validation.tf_record'  # 输出文件地址
ckpt_path = r"E:\pycharm_project\project_mtcnn\checkpoint\feature\feature"
pb_path = r"E:\pycharm_project\project_mtcnn\protocolbuffer\feature\class_model.pb"

font_path = r"E:\pycharm_project\dataset\font\simhei.ttf"

class Net:

    def __init__(self):

        self.x = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name="input")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.c = tf.Variable(tf.truncated_normal(shape=[10, 1024], stddev=tf.sqrt(1 / 1024), dtype=tf.float32))

        self.w1 = tf.Variable(tf.truncated_normal(shape=[7, 7, 3, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.b1 = tf.Variable(tf.zeros([64], dtype=tf.float32))  # [112*112*64]

        '''block 1'''
        self.block_w11 = tf.Variable(tf.truncated_normal(shape=[1,1,64,64],stddev=tf.sqrt(1/64),dtype=tf.float32))
        self.block_b11 = tf.Variable(tf.zeros(shape=[64],dtype=tf.float32))#56*56*64

        self.block_w12 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.block_b12 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))  # 56*56*64

        self.block_w13 = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.block_b13 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))  # 56*56*256

        self.inside_w1 = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 64], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.inside_b1 = tf.Variable(tf.zeros(shape=[64], dtype=tf.float32))  # 56*56*256

        self.up_convert_w1 = tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.up_convert_b1 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))  # 56*56*256

        self.down_convert_w1= tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.down_convert_b1 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))  # 28*28*512

        '''block 2'''
        self.block_w21 = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.block_b21 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))  # 28*28*128

        self.block_w22 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.block_b22 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))  # 28*28*128

        self.block_w23 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 512], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.block_b23 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))  # 28*28*512

        self.inside_w2 = tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.inside_b2 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float32))  # 56*56*512

        self.up_convert_w2 = tf.Variable(tf.truncated_normal(shape=[1, 1, 128, 512], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.up_convert_b2 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))  # 28*28*512

        self.down_convert_w2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 1024], stddev=tf.sqrt(1 / 1024), dtype=tf.float32))
        self.down_convert_b2 = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32))  # 14*14*1024

        '''block 3'''
        self.block_w31 = tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.block_b31 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))  # 14*14*256

        self.block_w32 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.block_b32 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))  # 14*14*256

        self.block_w33 = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 1024], stddev=tf.sqrt(1 / 1024), dtype=tf.float32))
        self.block_b33 = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32))  # 14*14*1024

        self.inside_w3 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1024, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.inside_b3 = tf.Variable(tf.zeros(shape=[256], dtype=tf.float32))  # 14*14*1024

        self.up_convert_w3 = tf.Variable(tf.truncated_normal(shape=[1, 1, 256, 1024], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.up_convert_b3 = tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32))  # 14*14*1024

        self.down_convert_w3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 1024, 2048], stddev=tf.sqrt(1 / 2048), dtype=tf.float32))
        self.down_convert_b3 = tf.Variable(tf.zeros(shape=[2048], dtype=tf.float32))  # 7*7*2048

        '''block 4'''
        self.block_w41 = tf.Variable(tf.truncated_normal(shape=[1, 1, 1024, 512], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.block_b41 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))  # 7*7*512

        self.block_w42 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.block_b42 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))  # 7*7*512

        self.block_w43 = tf.Variable(tf.truncated_normal(shape=[1, 1, 512, 2048], stddev=tf.sqrt(1 / 2048), dtype=tf.float32))
        self.block_b43 = tf.Variable(tf.zeros(shape=[2048], dtype=tf.float32))  # 7*7*2048

        self.inside_w4 = tf.Variable(tf.truncated_normal(shape=[1, 1, 2048, 512], stddev=tf.sqrt(1 / 512), dtype=tf.float32))
        self.inside_b4 = tf.Variable(tf.zeros(shape=[512], dtype=tf.float32))  # 14*14*1024

        self.fcn_w = tf.Variable(tf.truncated_normal(shape=[2048,1024], stddev=tf.sqrt(1 / 1024), dtype=tf.float32))
        self.fcn_b =  tf.Variable(tf.zeros(shape=[1024], dtype=tf.float32))  # 1024

        self.out_w = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=tf.sqrt(1 / 10), dtype=tf.float32))
        self.out_b = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))

    def building_block(self,input,w1,b1,w2,b2,w3,b3,strides):

        self.block_y1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(input,w1,[1,strides,strides,1],padding="SAME")+b1))
        self.block_y2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.block_y1,w2,[1,1,1,1],padding="SAME")+b2))
        self.block_y3 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.block_y2, w3, [1, 1, 1, 1], padding="SAME") + b3))

        return self.block_y3

    def forward(self):
        self.y1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.x, self.w1, [1, 2, 2, 1], padding="SAME") + self.b1))  # 112*112*64
        self.pool1 = tf.nn.max_pool(self.y1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")  # 56*56*64

        self.layers1_block1 = self.building_block(self.pool1, self.block_w11,self.block_b11,self.block_w12,self.block_b12,self.block_w13,self.block_b13,1)#56*56*256
        self.up_convert_y1 = tf.nn.conv2d(self.pool1,self.up_convert_w1,[1,1,1,1],padding="SAME")+self.up_convert_b1#56*56*256
        self.layers1_res_add1 = tf.nn.relu(self.up_convert_y1 + self.layers1_block1)  # 56*56*256

        self.layers1_block2 = self.building_block(self.layers1_res_add1,self.inside_w1,self.inside_b1,self.block_w12,self.block_b12,self.block_w13,self.block_b13,1)#56*56*256
        self.layers1_res_add2 = tf.nn.relu(self.layers1_res_add1 + self.layers1_block2)  # 56*56*256

        self.layers1_block3 = self.building_block(self.layers1_res_add2,self.inside_w1,self.inside_b1,self.block_w12,self.block_b12,self.block_w13,self.block_b13,1)#56*56*256
        self.layers1_res_add3 = tf.nn.relu(self.layers1_res_add2 + self.layers1_block3)  # 56*56*256
        self.layers1_res_add3_2 = tf.nn.conv2d(self.layers1_res_add3,self.down_convert_w1,[1,2,2,1],padding="SAME")+self.down_convert_b1#28*28*512

        self.layers2_block1 = self.building_block(self.layers1_res_add3,self.block_w21,self.block_b21,self.block_w22,self.block_b22,self.block_w23,self.block_b23,2)# 28*28*512
        self.layers2_res_add1 = tf.nn.relu(self.layers1_res_add3_2 + self.layers2_block1)  # 28*28*512

        self.layers2_block2 = self.building_block(self.layers2_res_add1, self.inside_w2,self.inside_b2,self.block_w22,self.block_b22,self.block_w23,self.block_b23,1)# 28*28*512
        self.layers2_res_add2 = tf.nn.relu(self.layers2_res_add1 + self.layers2_block2)  # 28*28*512

        self.layers2_block3 = self.building_block(self.layers2_res_add2, self.inside_w2,self.inside_b2,self.block_w22,self.block_b22,self.block_w23,self.block_b23,1)# 28*28*512
        self.layers2_res_add3 = tf.nn.relu(self.layers2_res_add2 + self.layers2_block3)  # 28*28*512

        self.layers2_block4 = self.building_block(self.layers2_res_add3, self.inside_w2,self.inside_b2,self.block_w22,self.block_b22,self.block_w23,self.block_b23,1)  # 28*28*512
        self.layers2_res_add4 = tf.nn.relu(self.layers2_res_add3 + self.layers2_block4)  # 28*28*512
        self.layers2_res_add4_2 = tf.nn.conv2d(self.layers2_res_add4, self.down_convert_w2, [1, 2, 2, 1],padding="SAME") + self.down_convert_b2  # 14*14*1024

        self.layers3_block1 = self.building_block(self.layers2_res_add4, self.block_w31, self.block_b31,self.block_w32,self.block_b32,self.block_w33,self.block_b33,2)  # 14*14*1024
        self.layers3_res_add1 = tf.nn.relu(self.layers2_res_add4_2 + self.layers3_block1)  # 14*14*1024

        self.layers3_block2 = self.building_block(self.layers3_res_add1, self.inside_w3,self.inside_b3,self.block_w32,self.block_b32,self.block_w33,self.block_b33,1)  # 14*14*1024
        self.layers3_res_add2 = tf.nn.relu(self.layers3_res_add1 + self.layers3_block2)  # 14*14*1024

        self.layers3_block3 = self.building_block(self.layers3_res_add2, self.inside_w3,self.inside_b3,self.block_w32,self.block_b32,self.block_w33,self.block_b33,1)  # 14*14*1024
        self.layers3_res_add3 = tf.nn.relu(self.layers3_res_add2 + self.layers3_block3)  # 14*14*1024

        self.layers3_block4 = self.building_block(self.layers3_res_add3, self.inside_w3,self.inside_b3,self.block_w32,self.block_b32,self.block_w33,self.block_b33,1)  # 14*14*1024
        self.layers3_res_add4 = tf.nn.relu(self.layers3_res_add3 + self.layers3_block4)  # 14*14*1024

        self.layers3_block5 = self.building_block(self.layers3_res_add4, self.inside_w3,self.inside_b3,self.block_w32,self.block_b32,self.block_w33,self.block_b33,1)  # 14*14*1024
        self.layers3_res_add5 = tf.nn.relu(self.layers3_res_add4 + self.layers3_block5)  # 14*14*1024

        self.layers3_block6 = self.building_block(self.layers3_res_add5, self.inside_w3,self.inside_b3,self.block_w32,self.block_b32,self.block_w33,self.block_b33,1)  # 14*14*1024
        self.layers3_res_add6 = tf.nn.relu(self.layers3_res_add5 + self.layers3_block6)  # 14*14*1024
        self.layers3_res_add6_2 = tf.nn.conv2d(self.layers3_res_add6, self.down_convert_w3, [1, 2, 2, 1],padding="SAME") + self.down_convert_b3  # 7*7*2048

        self.layers4_block1 = self.building_block(self.layers3_res_add6, self.block_w41, self.block_b41,self.block_w42,self.block_b42,self.block_w43,self.block_b43,2)  # 7*7*2048
        self.layers4_res_add1 = tf.nn.relu(self.layers3_res_add6_2 + self.layers4_block1)  # 7*7*2048

        self.layers4_block2 = self.building_block(self.layers4_res_add1,self.inside_w4,self.inside_b4,self.block_w42,self.block_b42,self.block_w43,self.block_b43,1)  # 7*7*2048
        self.layers4_res_add2 = tf.nn.relu(self.layers4_res_add1 + self.layers4_block2)  # 7*7*2048

        self.layers4_block3 = self.building_block(self.layers4_res_add2,self.inside_w4,self.inside_b4,self.block_w42,self.block_b42,self.block_w43,self.block_b43,1)  # 7*7*2048
        self.layers4_res_add3 = tf.nn.relu(self.layers4_res_add2 + self.layers4_block3)  # 7*7*2048
        self.pool2 = tf.nn.avg_pool(self.layers4_res_add3,ksize=[1,7,7,1],strides=[1,7,7,1],padding="SAME")#1*1*2048

        self.convert_y = tf.reshape(self.pool2,[-1,2048])#-1,2048

        self.c_y = tf.multiply(tf.matmul(self.convert_y,self.fcn_w)+self.fcn_b,1,name="center_feature")#-1,1024
        self.ys = tf.layers.batch_normalization(tf.matmul(self.c_y,self.out_w)+self.out_b)#-1,10
        self.output = tf.nn.softmax(self.ys)

    def backward(self):
        self.max_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.ys))
        self.center_loss = c_loss.center_loss(self.c_y, self.y, self.c)

        self.loss = self.max_loss+self.center_loss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    def accuracys(self):
        y = tf.equal(tf.argmax(self.output,axis=1),tf.argmax(self.y,axis=1))
        self.accuracy = tf.reduce_sum(tf.cast(y,dtype=tf.float32))

if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    net.accuracys()
    net.init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    plt.ion()
    a = []
    b = []
    train_get_batch = 5
    validate_get_batch = 1

    number = 0
    accur = 0
    numbers = 0
    accurs = 0
    ls = listes.lists()
    font = fonts.truetype(font_path, size=20)

    train_images, train_labels = sampling_feature_train_record.train.read_data_for_file(train_filename, 1,[224, 224, 3])  # 给出1张图片的初始化参数
    train_images_batch, train_labels_batch = tf.train.shuffle_batch([train_images, train_labels], batch_size=train_get_batch,capacity=5000, min_after_dequeue=100,num_threads=3)  # 打乱批次

    validation_images, validation_labels = sampling_feature_validation_record.test.read_data_for_file(validation_filename, 1, [224, 224, 3])  # 给出1张图片的初始化参数
    validation_images_batch, validation_labels_batch = tf.train.shuffle_batch([validation_images, validation_labels], batch_size=validate_get_batch, capacity=5000, min_after_dequeue=100,num_threads=3)  # 打乱批次

    with tf.Session() as sess:
        sess.run(net.init)
        # saver.restore(sess,ckpt_path)
        coord = tf.train.Coordinator()
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(999999999999999):
            xs, ys = sess.run([train_images_batch, train_labels_batch])
            ys = ys.astype(np.int32)
            ys = one_hot.one_hot_(train_get_batch,10,ys)
            output,center_error,max_loss,error,accuracy,_ = sess.run([net.output,net.center_loss,net.max_loss,net.loss,net.accuracy,net.optimizer],feed_dict={net.x:xs,net.y:ys})

            if i%10 == 0:
                xs_s,ys_s = sess.run([validation_images_batch, validation_labels_batch])
                ys_s = ys_s.astype(np.int32)
                ys_s = one_hot.one_hot_(validate_get_batch, 10, ys_s)
                _output, _accuracy = sess.run([net.output,net.accuracy], feed_dict={net.x: xs_s,net.y:ys_s})

                print("第",i,"次")
                mx = max(ys[0])
                yss = list(ys[0])
                label_cls = yss.index(mx)
                label_name = ls[label_cls]
                print("train_标签类别：",label_cls)

                mxo = max(output[0])
                ysso = list(output[0])
                out_cls = ysso.index(mxo)
                out_name = ls[out_cls]
                print("train_输出类别：",out_cls)

                mxlv = max(ys_s[0])
                ysslv = list(ys_s[0])
                label_vcls = ysslv.index(mxlv)
                label_vname = ls[label_vcls]
                print("validate_标签类别：",label_vcls)

                mxv = max(_output[0])
                yssv = list(_output[0])
                validate_cls = yssv.index(mxv)
                validate_name = ls[validate_cls]
                print("validate_输出类别：", validate_cls)

                number = number+train_get_batch
                accur = accur+accuracy
                print("train_error:", center_error,max_loss,error,'\n' "train_accuracy:",accur/number)

                numbers = numbers + validate_get_batch
                accurs = accurs + _accuracy
                print("validate_accuracy:", accurs / numbers)
                accuracys = accurs / numbers

                # print(ys[0])
                # print(output[0])
                # print(ys_s[0])
                # print(_output[0])

                img = xs[0]*255
                image = pimg.fromarray(np.uint8(img))
                imgdraw = pdraw.ImageDraw(image)
                imgdraw.text(xy=(0, 0), text=("我是train标签:" + label_name), fill="blue", font=font)
                imgdraw.text(xy=(0, 30), text=("我是train输出:" + out_name), fill="red", font=font)
                # image.show()

                img = xs_s[0] * 255
                image = pimg.fromarray(np.uint8(img))
                imgdraw = pdraw.ImageDraw(image)
                imgdraw.text(xy=(0, 0), text=("我是validate标签:" + label_vname), fill="blue", font=font)
                imgdraw.text(xy=(0, 30), text=("我是validate输出:" + validate_name), fill="red", font=font)
                # image.show()

                # a.append(i)
                # b.append(error)
                # plt.clf()
                # plt.plot(a,b)
                # plt.pause(0.001)

                # saver.save(sess,ckpt_path)
                # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["input","center_feature"])
                # with tf.gfile.FastGFile(pb_path, mode='wb') as f:
                #     f.write(constant_graph.SerializeToString())
                if accuracys >0.72 and error < 0.035 :
                    break