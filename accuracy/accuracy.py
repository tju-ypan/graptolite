import tensorflow as tf

batch_size = 8
class_num = 10  # 类别数量

# 定义一个logits为神经网络预测的标签结果，shape:(batch_size, )
logits = tf.constant([0, 5, 9, 1, 7, 1, 0, 1])
# 定义一个labels为真实样本号，这里设为全1，shape:(batch_size, )
labels = tf.ones((batch_size, ), dtype=tf.int32)

# 使用tf.metrics.accuracy()计算分类准确率，返回的第一个值即为分类准确率
acc, acc_op = tf.metrics.accuracy(logits, labels)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print(logits.eval())
    print(labels.eval())
    print("accuracy:{}".format(acc_op.eval()))