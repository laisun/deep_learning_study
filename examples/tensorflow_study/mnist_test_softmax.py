#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os,sys
import tensorflow as tf
import input_data

# read imagedata
mnist = input_data.read_data_sets("./tmp/", one_hot=True)


# define variables
x = tf.placeholder(tf.float32, [None, 784])  #input features: x 
y_ = tf.placeholder("float", [None,10])      #input lables : y

W = tf.Variable(tf.zeros([784,10]))  # weighted matrix
b = tf.Variable(tf.zeros([10]))      # bias vector

# define model
y = tf.nn.softmax(tf.matmul(x,W) + b)

# define cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# define Optimizer
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# model evaluation
# argmax:the index of the biggest value in the vector.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# print the result
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


