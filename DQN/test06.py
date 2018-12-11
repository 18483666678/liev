import tensorflow as tf

a = tf.Variable([[0,0],[0,0]])
c = tf.assign_add(a,[[1,1],[1,1]])
d = tf.summary.histogram('d',a)  #收集直方图

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs',sess.graph)
    sess.run(init)

    for i in range(100):
        print(sess.run(c))
        summary = sess.run(merged)
        writer.add_summary(summary,i)