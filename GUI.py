# !/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import tensorflow as tf
import logging
import matplotlib

matplotlib.use('TkAgg')

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from PIL import Image
from tkinter import Tk, Frame, Checkbutton, StringVar, Button
from tkinter import BooleanVar, BOTH, LEFT, BOTTOM, N, S, RIGHT, E

logger = logging.getLogger('Training a chiness write char recognition')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_integer('image_width', 28,
                            "the width of image ")
tf.app.flags.DEFINE_integer('image_height', 28, 'the height of image')
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

        return 1 - np.mean([r_arr, g_arr, b_arr], axis=0) / 255

    def next_batch(self, batch_size):
        arr = []
        for i in range(batch_size):
            arr.append(random.randint(0, len(self.image_names) - 1))
        x = []
        y = []
        for i in range(batch_size):
            y.append(self.labels[i])
            x.append(self.image_to_array(self.image_names[i]))
        return x, y

    def all(self):
        x = []
        y = []
        for i in range(len(self.image_names)):
            y.append(self.labels[i])
            x.append(self.image_to_array(self.image_names[i]))
        return x, y

    def select_a_random_image(self):
        index = random.randint(0, len(self.image_names) - 1)
        filename = self.image_names[index]
        return Image.open(filename), filename


class App(Frame):
    sess = None

    def __init__(self, window, **kw):
        Frame.__init__(self, window)
        self.window = window
        self.window.title("Letter Recognition")

        frame = Frame(window)
        frame.pack(side=LEFT, fill=BOTH)
        Button(frame, text="Step Training", command=lambda: self.train(1)).pack(fill=BOTH, padx=5, pady=5)
        self.start_button_name = StringVar(value="Start Training")
        Button(frame, textvariable=self.start_button_name, command=self.start).pack(fill=BOTH, padx=5, pady=5)
        Button(frame, text="Clear & Reset", command=self.init_model).pack(fill=BOTH, padx=5, pady=5)
        Button(frame, text="Test", command=self.test).pack(fill=BOTH, padx=5, pady=5)
        Button(frame, text="Quit", command=window.quit).pack(side=BOTTOM, fill=BOTH, padx=5, pady=5)
        self.is_V2 = BooleanVar(value=False)
        Checkbutton(frame, text="Use 2 layers (Unstable)", variable=self.is_V2, command=self.init_model).pack(fill=BOTH,
                                                                                                              padx=5,
                                                                                                              pady=5)
        self.stop = True
        self.fig = Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=True)

        self.subplots = []
        self.subplots.append(self.fig.add_subplot(231))
        self.subplots.append(self.fig.add_subplot(232))
        self.subplots.append(self.fig.add_subplot(233))
        self.fig.subplots_adjust(right=0.8)
        self.cbar = self.fig.add_axes([0.85, 0.5, 0.05, 0.4])

        self.subplot = self.fig.add_subplot(223)
        self.test_image = self.fig.add_subplot(224)
        self.subplot.set_title("Accuracy Diagram", fontsize=16)
        self.subplot.set_ylabel("Accuracy")
        self.subplot.set_xlabel("Step")

        # canvas = Canvas(self, width=10, height=10)
        # canvas.pack(side=TOP)
        #
        # pilImage = Image.new("RGB", (500, 500), 0)
        # image = ImageTk.PhotoImage(pilImage)
        # imagesprite = canvas.create_image(400, 400, image=image)

        self.train_feeder = DataIterator(data_dir=FLAGS.train_data_dir)
        self.test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)

        self.test_batch_xs, self.test_batch_ys = self.test_feeder.all()

        self.init_model()

    def init_model(self):
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.InteractiveSession()

        self.count = 0
        self.subplot.clear()
        for subplot in self.subplots:
            subplot.clear()

        self.canvas.draw()

        self.plot_x = [0]
        self.plot_y = [0]
        if self.is_V2.get():
            # Create the model
            self.x = tf.placeholder(tf.float32, [None, 28 * 28])
            self.y_ = tf.placeholder(tf.float32, [None, 26])

            x_image = tf.reshape(self.x, [-1, 28, 28, 1])

            # conv1
            W_conv1 = weight_variable([5, 5, 1, 32])  # 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目 (1 #28x28->32 #28x28)
            b_conv1 = bias_variable([32])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)  # (32 #28x28->32 #14x14)

            # conv2
            W_conv2 = weight_variable([5, 5, 32, 64])  # (32 #14x14->64 #14x14->64 #7x7)
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            self.keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            W_fc2 = weight_variable([1024, 26])
            b_fc2 = bias_variable([26])

            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            # test
            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            # Create the model
            self.x = tf.placeholder(tf.float32, [None, 28 * 28])
            self.y_ = tf.placeholder(tf.float32, [None, 26])

            self.W = tf.Variable(tf.zeros([28 * 28, 26]))
            b = tf.Variable(tf.zeros([26]))
            self.y = tf.matmul(self.x, self.W) + b

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
            # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

            # test
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess.run(tf.global_variables_initializer())

    def train(self, times):
        self.count += times
        with self.sess.as_default():
            for i in range(times):
                batch_xs, batch_ys = self.train_feeder.next_batch(100)
                if self.is_V2.get():
                    self.train_step.run(feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
                else:
                    self.train_step.run(feed_dict={self.x: batch_xs, self.y_: batch_ys})

            if self.is_V2.get():
                train_accuracy = self.accuracy.eval(
                    feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
                test_accuracy = self.accuracy.eval(
                    feed_dict={self.x: self.test_batch_xs, self.y_: self.test_batch_ys, self.keep_prob: 1.0})
            else:
                train_accuracy = self.accuracy.eval(feed_dict={self.x: batch_xs, self.y_: batch_ys})
                test_accuracy = self.accuracy.eval(
                    feed_dict={self.x: self.test_batch_xs, self.y_: self.test_batch_ys})

        print(
            "Step %d, training accuracy %g, test accuracy %g" % (
                self.count, train_accuracy, test_accuracy))
        self.plot_x.append(self.count)
        self.plot_y.append(test_accuracy)
        self.plot()

    def plot(self):
        self.subplot.axis([0, self.plot_x[-1] + 5, 0, 1])

        self.subplot.plot(self.plot_x, self.plot_y, color='blue')

        if not self.is_V2.get():
            W = []
            for i in range(3):
                W.append(self.sess.run(tf.reshape(tf.transpose(self.W)[i], [28, 28])))
                im = self.subplots[i].imshow(W[i], vmin=-0.15, vmax=0.15)
            self.fig.colorbar(im, cax=self.cbar)

        self.canvas.draw()

    def start(self):
        if self.stop:
            self.start_button_name.set("Stop")
            self.stop = False
            threading.Timer(0, self.work).start()
        else:
            self.stop = True

    def work(self):
        if self.stop:
            self.start_button_name.set("Start Training")
            return
        self.train(5)
        threading.Timer(0, self.work).start()

    def test(self):
        image, name = self.test_feeder.select_a_random_image()
        self.test_image.imshow(np.array(image))
        self.canvas.draw()
        print("Loaded an image: " + name)
        image_array = [self.test_feeder.image_to_array(name)]

        if self.is_V2.get():
            results = self.sess.run(tf.nn.softmax(self.y_conv), feed_dict={self.x: image_array, self.keep_prob: 1.0})[0]
        else:
            results = self.sess.run(tf.nn.softmax(self.y), feed_dict={self.x: image_array})[0]

        combine_list = list()
        for i in range(26):
            combine_list.append((chr(65 + i), results[i]))
        combine_list.sort(key=lambda x: x[1], reverse=True)

        string = ""
        for i in range(3):
            string += combine_list[i][0] + ":" + str(combine_list[i][1]) + "\t"

        print("Result: " + string)


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


def main():
    window = Tk()
    window.geometry("500x400+300+300")
    start = App(window)
    window.mainloop()


if __name__ == '__main__':
    main()
