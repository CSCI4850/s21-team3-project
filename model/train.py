from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from matplotlib import cm

import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import pathlib
import matplotlib.pyplot as plt
import pickle

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


class Model:
    """
        Model Class for Model Abstraction, Class implentation is similar to the 
        Tensorflow Sequential and Functional Model Using Graph 
        Please refer to: https://www.tensorflow.org/api_docs/python/tf/Graph for more details
    """
    def __init__(self, commands, input_size=1960, first_conv_filter =128, second_conv_filter=64, model_dir="model", frequency_size = 40, time_size=49, sess=False):
        """
            Initialization of variables and Tensor Session
        """
        self.check_session(sess)
        self.commands = commands
        self.commands_dic = self.create_commands(self.commands)
        self._softmax_layer, self._dropout_placeholder = self._build(input_size, first_conv_filter, second_conv_filter, frequency_size, time_size)
        self._model_dir = model_dir
        self._input_size = input_size
        self._loaded = False
        self._start_step = 0
        self._global_step = tf.compat.v1.train.get_or_create_global_step()

        assert type(commands) == list, " Commands type should be a list "
        assert type(model_dir) == str, "model directory should be a string object"

        
        


        
    def _build(self, input_size, first_conv_filter, second_conv_filter, frequency_size, time_size):

        """
            This a private protected Method to Build the Model Layer in graph
            
            Args:
                input_size: Size of the flattened input default to 1960
                first_conv_filter : Size of filter for first convolutional layer
                second_conv_filter : Size of filter for second convolutional layer
                frequecncy_size : Size of MFCC rows. Refer to feature extraction for run_MFCC method
                time_size : Size of MFCC cols
            returns:
                Returns are abstracted

        """

        dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
        
        
        self._fingerprint_input = tf.compat.v1.placeholder(
            tf.float32, [None, input_size], name='fingerprint_input')


        input_4d = tf.reshape(self._fingerprint_input,                      # input: MFCC for commands [batch_size, input_size]
                                    [-1, time_size, frequency_size, 1])     # output reshape [batch_size, rows, cols, channel]

        first_weights = tf.compat.v1.get_variable(                          # Weights Initialization 
            name='first_weights',
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            shape=[20, 8, 1, first_conv_filter])

        first_bias = tf.compat.v1.get_variable(                              # Bias Initialization 
            name='first_bias',
            initializer=tf.compat.v1.zeros_initializer,
            shape=[first_conv_filter,])


        first_conv = tf.nn.conv2d(input=input_4d,                         # First Convolution Layer
                                filters=first_weights,                    #input: [batch_size, rows, cols, channel]
                                strides=[1, 1, 1, 1],                     #output: [20, 8, 1, first_filter_count]
                                padding='SAME') + first_bias


        first_relu = tf.nn.relu(first_conv)

        first_dropout = tf.nn.dropout(first_relu, rate=dropout_rate)

        max_pool = tf.nn.max_pool2d(input=first_dropout,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
        
        second_weights = tf.compat.v1.get_variable(
            name='second_weights',
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            shape=[10, 4, first_conv_filter, second_conv_filter])

        second_bias = tf.compat.v1.get_variable( name='second_bias',
            initializer=tf.compat.v1.zeros_initializer,
            shape=[second_conv_filter,])

        second_conv = tf.nn.conv2d(input=max_pool,
                                    filters=second_weights,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME') + second_bias
        second_relu = tf.nn.relu(second_conv)
        second_dropout = tf.nn.dropout(second_relu, rate=dropout_rate)


        conv_shape = second_dropout.get_shape()
        conv_output_width = conv_shape[2]
        conv_output_height = conv_shape[1]

        conv_element_count = int(
            conv_output_width * conv_output_height * second_conv_filter)

        flattened_second_conv = tf.reshape(second_dropout,
                                            [-1, conv_element_count])
        label_count = len(self.commands_dic)

        softmax_weights = tf.compat.v1.get_variable(
            name='softmax_weights',
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01),
            shape=[conv_element_count, label_count])


        softmax_bias = tf.compat.v1.get_variable( name='softmax_bias',
            initializer=tf.compat.v1.zeros_initializer,
            shape=[label_count])


        softmax_layer = tf.matmul(flattened_second_conv, softmax_weights) + softmax_bias

        return softmax_layer, dropout_rate

    def train(self, learn_rate, dropout_rate, save_step, batch_size, training_time, rate_step, display_step, train_data, Validation_data):
        assert type(learn_rate) == list,\
             "Learn Rate should be a List to be used. e.g [.001, .0001]"
        
        self._ground_truth_input = tf.compat.v1.placeholder(
            tf.int64, [None], name='groundtruth_input')

        with tf.compat.v1.name_scope('cross_entropy'):
            self._cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                labels=self._ground_truth_input, logits=self._softmax_layer)


        learning_rate_input = tf.compat.v1.placeholder(
                tf.float32, [], name='learning_rate_input')

        train_step = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate_input).minimize(self._cross_entropy_mean)

        self._predicted = tf.argmax(input=self._softmax_layer, axis=1)
        correct_prediction = tf.equal(self._predicted, self._ground_truth_input)
    
        self._evaluation_step = tf.reduce_mean(input_tensor=tf.cast(correct_prediction,
                                                                tf.float32))

        
        

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        
        if not self._loaded:
            self._global_step = tf.compat.v1.train.get_or_create_global_step()
            tf.compat.v1.global_variables_initializer().run()
            self._loaded = True

        increment_global_step = tf.compat.v1.assign(self._global_step, self._global_step + 1)
        tf.io.write_graph(self._sess.graph_def, self._model_dir, "model"+ '.pbtxt')
        
        with gfile.GFile(
        os.path.join(self._model_dir, "commands" + '_labels.txt'),'wb') as f:
            f.write('\n'.join(self.commands))

        if training_time <= self._start_step and self._loaded:
            print(f"Checkpoint Loaded has been trained to {self._start_step},\
                New Trainig starts from {self._start_step}, Please increase Training_time to train model")

        strategy = strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            history = {
                "categorical_accuracy":[],
                "loss": [],
                "val_categorical_accuracy":[],
                "val_loss":[]}
            learning_rate = learn_rate[0]
            for training_step in xrange(self._start_step, training_time):
                if training_step == int(rate_step):
                    learning_rate = learn_rate[1]

                x_train, y_train = self.get_train_batch(batch_size, train_data)
                train_accuracy, cross_entropy_value, _, _ = self._sess.run(
                    
                    [
                        self._evaluation_step,
                        self._cross_entropy_mean,
                        train_step,
                        increment_global_step,
                    ],
                    feed_dict={
                        self._fingerprint_input: x_train,
                        self._ground_truth_input: y_train,
                        learning_rate_input: learning_rate,
                        self._dropout_placeholder: dropout_rate
                    })
                history["categorical_accuracy"].append(train_accuracy)
                history["loss"].append(cross_entropy_value)
                if training_step % int(display_step) ==0:
                    print(
                        'Step #%d: learning rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate, train_accuracy * 100,
                        cross_entropy_value))

                x_val, y_val = self.get_train_batch(batch_size, Validation_data)
                validation_accuracy, val_crossentropy_value = self._sess.run(
                        [
                            self._evaluation_step, 
                            self._cross_entropy_mean
                        
                        ],
                        feed_dict={
                            self._fingerprint_input: x_val,
                            self._ground_truth_input: y_val,
                            self._dropout_placeholder: 0.0
                        })

                history["val_categorical_accuracy"].append(validation_accuracy)
                history["val_loss"].append(val_crossentropy_value)

                if training_step % int(display_step) ==0:
                    print('Step %d: Validation accuracy = %.1f%% (N=%d), Validation loss = %f' %
                                (training_step, validation_accuracy * 100, batch_size, val_crossentropy_value))

                if (training_step% int(save_step) ==0)or (training_step == training_time-1):
                    path_to_save = os.path.join(self._model_dir, "model_checkpoint" + '.ckpt')
                    saver.save(self._sess, path_to_save, global_step=training_step)

        return history


    def check_session(self, sess= False):
        if sess != False:
            if sess._closed:   
                if tf.test.is_built_with_cuda():  # Check GPU compatibility
                    from tensorflow.compat.v1 import ConfigProto
                    from tensorflow.compat.v1 import InteractiveSession

                    config = ConfigProto()
                    config.gpu_options.allow_growth = True
                    #sess.close()
                    self._sess = InteractiveSession(config=config)
                else:                            # Run on CPU if GPU is not available
                    #sess.close()
                    self._sess = InteractiveSession()
            else:
                self._sess = sess
        else:
            if tf.test.is_built_with_cuda():  # Check GPU compatibility
                from tensorflow.compat.v1 import ConfigProto
                from tensorflow.compat.v1 import InteractiveSession

                config = ConfigProto()
                config.gpu_options.allow_growth = True
                #sess.close()
                self._sess = InteractiveSession(config=config)
            else:                            # Run on CPU if GPU is not available
                #sess.close()
                self._sess = InteractiveSession()

    def create_commands(self, commands):
        commands_dic = {}
        for i in range(len(commands)):
            commands_dic[i] = commands[i]

        return commands_dic


    def get_train_batch(self, batch_size, dataset):
        np.random.shuffle(dataset)
        data = dataset[:batch_size, 0]
        label = dataset[:batch_size, 1]
        return np.stack(data), np.stack(label)

    
    def predict(self, input_data):
        


        predicted = self._sess.run(
            [self._predicted],
            feed_dict = {
                self._fingerprint_input: input_data,
                self._dropout_placeholder : 0.0

            }
        )
        
        return predicted[0], [self.commands_dic[n.item()] for n in predicted[0]]
       

    def evaluate(self, input_data, labels, verbose= 1):

        validation_accuracy, val_crossentropy_value = self._sess.run(
                        [self._evaluation_step, self._cross_entropy_mean],
                        feed_dict={
                            self._fingerprint_input: input_data,
                            self._ground_truth_input: labels,
                            self._dropout_placeholder: 0.0
                        })
        if verbose:
            print('Validation accuracy = %.1f%%, Validation loss = %f' %
                                (validation_accuracy * 100, val_crossentropy_value))
            
        return validation_accuracy, val_crossentropy_value


    def load_checkpoint(self, path=0):

        if path==0:
            path = os.path.join(self._model_dir, "model_checkpoint" + '.ckpt-0')
        
        #assert os.path.exists(path), "Path does not exist"

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(self._sess, path)
        self._start_step = self._global_step.eval(session=self._sess)
        self._loaded = True
        return True


    #def save_pb_model(self, path, filename):
        

