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
            if chr(65 + i) == char[0]:
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


if __name__ == '__main__':
    # Import data

    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    W = tf.Variable(tf.zeros([28 * 28, 26]))
    b = tf.Variable(tf.zeros([26]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 26])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # test

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = train_feeder.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            test_batch_xs, test_batch_ys = test_feeder.next_batch(26)
            # Test trained model
            print 'stage {0}/{1}'.format(i, FLAGS.max_steps)
            print 'accuracy {0}'.format(sess.run(accuracy, feed_dict={x: test_batch_xs,
                                                                      y_: test_batch_ys}))
            '''
            # 查看返回值为28*28的权重矩阵
            sess.run(tf.reshape(tf.transpose(W)[字母编号0-25], [28,28]))
            '''
