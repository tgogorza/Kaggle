import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing import smooth, denoise, color_composite, process_input

def get_data(data_path):
    with open(data_path, 'rb') as f:
        data = json.load(f)
    print 'Loaded {} items'.format(len(data))
    return pd.DataFrame(data)

df = get_data('data/train.json')
df.inc_angle = df.inc_angle.replace('na', 0)
df.inc_angle = df.inc_angle.astype(float).fillna(0.0)

X, Y = process_input(df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y.squeeze(), random_state=123, train_size=0.99)

learning_rate = 0.001
reg_param = 0.001
epochs = 30
batch_size = 25
display_step = 2

tf.reset_default_graph()
tf.set_random_seed(123)
np.random.seed(123)
x = tf.placeholder(tf.float32, shape=(None, X.shape[1], X.shape[2], X.shape[3]), name='x')
y = tf.placeholder(tf.int32, shape=(None), name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with tf.name_scope('ConvGraph'):
    with tf.name_scope('ConvLayer1'):
        conv1 = tf.layers.conv2d(x,
                                 filters=128,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=123),
                                 activation=tf.nn.relu, name='Convolution-5x5-128-channels')
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='MaxPool-2x2')
    with tf.name_scope('ConvLayer2'):
        conv2 = tf.layers.conv2d(pool1,
                                 filters=128,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=456),
                                 activation=tf.nn.relu,
                                 name='Convolution-3x3-128-channels')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='MaxPool-2x2')
    with tf.name_scope('ConvLayer3'):
        conv3 = tf.layers.conv2d(pool2,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=789),
                                 activation=tf.nn.relu,
                                 name='Convolution-3x3-256-channels')
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='MaxPool-2x2')
    with tf.name_scope('ConvLayer4'):
        conv4 = tf.layers.conv2d(pool3,
                                 filters=512,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=789),
                                 activation=tf.nn.relu,
                                 name='Convolution-3x3-512-channels')
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=[4, 4], strides=4, name='MaxPool-4x4')
    with tf.name_scope('FullyConnected'):
        pool4_flat = tf.contrib.layers.flatten(pool4)
        dropout0 = tf.layers.dropout(pool4_flat, keep_prob)
        dense1 = tf.layers.dense(dropout0, units=1024, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(dense1, keep_prob)
        dense2 = tf.layers.dense(dropout1, units=256, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(dense2, keep_prob)
        dense3 = tf.layers.dense(dropout2, units=64, activation=tf.nn.relu)
        dropout3 = tf.layers.dropout(dense3, keep_prob)
    with tf.name_scope('Output'):
        logits = tf.layers.dense(inputs=dropout3, units=2)
        tf.summary.tensor_summary('logits', logits)
        activ = tf.sigmoid(logits, name='activ')
        tf.summary.tensor_summary('activation', activ)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
    tf.summary.scalar("loss", cost)
    # cost_no_reg = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, reduction=tf.losses.Reduction.NONE)
    # regularizers = tf.nn.l2_loss(dense2) + tf.nn.l2_loss(dense3)
    # tf.summary.scalar("loss-no-reg", cost_no_reg)
    # cost = tf.reduce_mean(cost_no_reg + reg_param * regularizers)

with tf.name_scope("optimize"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('acc'):
    correct = tf.nn.in_top_k(activ, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./train', graph=sess.graph)
    test_writer = tf.summary.FileWriter('./test', graph=sess.graph)
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(len(X_train) / batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(Y_train, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            summary, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0 or epoch == epochs-1:
            print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(avg_cost))
            summary, acc_train = sess.run([merged, accuracy], feed_dict={x: X_train[:100], y: Y_train[:100], keep_prob: 1.0})
            train_writer.add_summary(summary, epoch)
            summary, acc_test = sess.run([merged, accuracy], feed_dict={x: X_test, y: Y_test, keep_prob: 1.0})
            test_writer.add_summary(summary, epoch)
            train_writer.flush()
            test_writer.flush()
            print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_test)
    print("Optimization Finished!")

    saver = tf.train.Saver()
    saver.save(sess, './conv1/conv5')
    print("Model saved")
