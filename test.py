# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
from PIL import Image
import logging

logger = logging.getLogger('Training a chiness write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh = logging.FileHandler('recogniiton.log')
# fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# logger.addHandler(fh)
logger.addHandler(ch)

tf.app.flags.DEFINE_integer('image_width', 28,
                            "the width of image ")
tf.app.flags.DEFINE_integer('image_height', 28, 'the height of image')
tf.app.flags.DEFINE_integer('max_steps', 2000, 'the max training steps ')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'the checkpoint dir')
''' 训练集目录  '''
tf.app.flags.DEFINE_string('train_data_dir', './images/train', 'the train dataset dir')
''' 测试集目录 '''
tf.app.flags.DEFINE_string('test_data_dir', './images/test', 'the test dataset dir')

FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                if file_path.split('.')[-1] == 'jpeg':
                    self.image_names += [os.path.join(root, file_path)]
        random.shuffle(self.image_names)

        self.labels = []
        for file_name in self.image_names:
            self.labels.append(self.to_label([file_name.split('/')[-1].split('.')[0].split('_')[0]]))

    def to_label(self, char):
        result = []
        for i in range(26):
            # for i in range(10):
            if chr(65 + i) == char[0]:
                # if str(i) == char[0]:
                flag = 1
            else:
                flag = 0
            result.append(flag)
        return result

    def image_to_array(self, filename):
        image = Image.open(filename)
        r, g, b = image.split()

        size = FLAGS.image_height * FLAGS.image_width

        r_arr = np.array(r).reshape(size)
        g_arr = np.array(g).reshape(size)
        b_arr = np.array(b).reshape(size)
        # r_arr = np.array(r)
        # g_arr = np.array(g)
        # b_arr = np.array(b)

        return np.mean([r_arr, g_arr, b_arr], axis=0)

    def next_batch(self, batch_size):
        arr = []
        for i in range(batch_size):
            arr.append(i)
        random.shuffle(arr)
        x = []
        y = []
        for i in range(batch_size):
            y.append(self.labels[i])
            x.append(self.image_to_array(self.image_names[i]))

        return x, y


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

is_V2 = True

if __name__ == '__main__':
    # Import data

    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)
    if is_V2:
        # Create the model
        x = tf.placeholder(tf.float32, [None, 28 * 28])
        y_ = tf.placeholder(tf.float32, [None, 26])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # conv1
        W_conv1 = weight_variable([5, 5, 1, 32])  # 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目 (1 #28x28->32 #28x28)
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1) # (32 #28x28->32 #14x14)

        # conv2
        W_conv2 = weight_variable([5, 5, 32, 64]) # (32 #14x14->64 #14x14->64 #7x7)
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 26])
        b_fc2 = bias_variable([26])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # test
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    else:
        # Create the model
        x = tf.placeholder(tf.float32, [None, 28 * 28])
        y_ = tf.placeholder(tf.float32, [None, 26])

        W = tf.Variable(tf.zeros([28 * 28, 26]))
        b = tf.Variable(tf.zeros([26]))
        y = tf.matmul(x, W) + b

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # test
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    test_batch_xs, test_batch_ys = test_feeder.next_batch(26)

    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = train_feeder.next_batch(100)
        if i % 5 == 0:
            if is_V2:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={x: test_batch_xs, y_: test_batch_ys, keep_prob: 1.0})
            else:
                train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
                test_accuracy = accuracy.eval(feed_dict={x: test_batch_xs, y_: test_batch_ys})

            print(
                "step %d/%d, training accuracy %g, test accuracy %g" % (
                i, FLAGS.max_steps, train_accuracy, test_accuracy))
        if is_V2:
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        else:
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    print("test accuracy %g" % accuracy.eval(feed_dict={ x: test_batch_xs, y_: test_batch_ys, keep_prob: 1.0}))

    sess.close()

# 查看返回值为28*28的权重矩阵
# sess.run(tf.reshape(tf.transpose(W)[字母编号0-25], [28,28]))