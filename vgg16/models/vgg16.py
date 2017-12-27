# ECE 544 Project vgg16 model
# fwu11
import numpy as np
import tensorflow as tf


class Vgg16(object):
    def __init__(self, rows, cols):

        # Create session
        self.session = tf.InteractiveSession()
        self.x_placeholder = tf.placeholder(tf.float32, [None, rows, cols, 3])
        self.y_placeholder = tf.placeholder(tf.float32, [None, 10])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.keep_prob_placeholder = tf.placeholder(tf.float32, [])

        # A boolean indicating whether the current mode is 'training'.
        self.phase_train = tf.placeholder(tf.bool,[])

        # Build graph.
        self.outputs_tensor, self.output_logits = self.build(
            self.x_placeholder, self.keep_prob_placeholder)

        # Setup loss tensor, accuracy_tensor, update_op_tensor
        self.loss_tensor = self.loss(self.output_logits, self.y_placeholder)
        self.accuracy_tensor = self.accuracy(
            self.outputs_tensor, self.y_placeholder)
        self.update_op_tensor = self.update_op(self.loss_tensor,
                                               self.learning_rate_placeholder)
        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())

    def build(self, X, keep_prob):

        print("build model started")
        #print(X.shape)
        with tf.variable_scope('model'):
            if self.phase_train == False:
                tf.get_variable_scope().reuse_variables()

            conv1_1 = self.conv_layer(
                X, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
            conv1_2 = self.conv_layer(
                conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
            pool1 = self.max_pool(conv1_2, name='pool1',
                                  kh=2, kw=2, dw=2, dh=2)

            conv2_1 = self.conv_layer(
                pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
            conv2_2 = self.conv_layer(
                conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
            pool2 = self.max_pool(conv2_2, name='pool2',
                                  kh=2, kw=2, dw=2, dh=2)

            conv3_1 = self.conv_layer(
                pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
            conv3_2 = self.conv_layer(
                conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
            conv3_3 = self.conv_layer(
                conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)
            pool3 = self.max_pool(conv3_3, name='pool3',
                                  kh=2, kw=2, dw=2, dh=2)

            conv4_1 = self.conv_layer(
                pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv4_2 = self.conv_layer(
                conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv4_3 = self.conv_layer(
                conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
            pool4 = self.max_pool(conv4_3, name='pool4',
                                  kh=2, kw=2, dw=2, dh=2)

            conv5_1 = self.conv_layer(
                pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv5_2 = self.conv_layer(
                conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
            conv5_3 = self.conv_layer(
                conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
            pool5 = self.max_pool(conv5_3, name='pool5',
                                  kh=2, kw=2, dw=2, dh=2)

            shp = pool5.get_shape()
            flattened_shape = shp[1].value * shp[2].value * shp[3].value
            resh = tf.reshape(pool5, [-1, flattened_shape], name='resh')

            fc6 = self.fc_layer(resh, name="fc6", n_out=4096)
            relu6 = tf.nn.relu(fc6, name='relu6')
            fc6_drop = tf.nn.dropout(relu6, keep_prob, name='drop6')

            fc7 = self.fc_layer(fc6_drop, name="fc7", n_out=4096)
            relu7 = tf.nn.relu(fc7, name='relu7')
            fc7_drop = tf.nn.dropout(relu7, keep_prob, name='drop7')

            logits = self.fc_layer(fc7_drop, name="fc8", n_out=10)

            prob = tf.nn.softmax(logits, name="prob")

        return prob, logits

    def max_pool(self, input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

    def conv_layer(self, input_op, name, kh, kw, n_out, dh, dw):
        n_in = input_op.get_shape()[-1].value

        with tf.variable_scope(name):
            W = tf.get_variable('weights', shape=[
                                kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('bias', shape=[
                                n_out], dtype=tf.float32, initializer=tf.constant_initializer())
            conv = tf.nn.conv2d(input_op, W, strides=[
                                1, dh, dw, 1], padding='SAME') + B
            activation = tf.nn.relu(conv)

        return activation

    def fc_layer(self, input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value
        with tf.variable_scope(name):
            W = tf.get_variable('weights', shape=[
                                n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            B = tf.get_variable('bias', shape=[
                                n_out], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
            fc = tf.nn.bias_add(tf.matmul(input_op, W),B)

        return fc

    def loss(self, y_logits, y_label):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_label, logits=y_logits))
        return cross_entropy_loss

    def accuracy(self, f, y):
        correct_prediction = tf.equal(tf.cast(tf.argmax(f, 1), tf.float32), tf.cast(tf.argmax(y, 1), tf.float32))
        prediction_accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        return prediction_accuracy

    def update_op(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer
