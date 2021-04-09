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


def load_wav_file(filename, sess, sample_rate=16000):
    """Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
     filename: Path to the .wav file to load.

    Returns:
     Numpy array holding the sample data as floats between -1.0 and 1.0.
    """
  
    wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=sample_rate)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()


def save_wav_file(filename, wav_data, sess, sample_rate=16000):
  """Saves audio sample data to a .wav audio file.

  Args:
    filename: Path to save the file to.
    wav_data: 2D array of float PCM-encoded audio data.
    sample_rate: Samples per second to encode in the file.
  """
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
    sample_rate_placeholder = tf.compat.v1.placeholder(tf.int32, [])
    wav_data_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 1])
    wav_encoder = tf.audio.encode_wav(wav_data_placeholder,
                                      sample_rate_placeholder)
    wav_saver = io_ops.write_file(wav_filename_placeholder, wav_encoder)
    sess.run(
        wav_saver,
        feed_dict={
            wav_filename_placeholder: filename,
            sample_rate_placeholder: sample_rate,
            wav_data_placeholder: np.reshape(wav_data, (-1, 1))
        })
    
def get_spectrogram(filename, window_size_samples, window_stride_samples, sess):
    """Create Spectrogram from the PCM-encoded audio data

     Args:
        wav_data: 2D array of float PCM-encoded audio data.
        sess: current session being run
     Returns:
         2-D spectrogram of audio
     
      """
    wav_data = load_wav_file(filename, sess)
    #print(wav_data.shape)
    wav_data_placeholder = tf.compat.v1.placeholder(tf.float32, [None, 1])
    spectrogram = audio_ops.audio_spectrogram(
          wav_data_placeholder,
          window_size=window_size_samples,
          stride=window_stride_samples,
          magnitude_squared=True)

    spectrogram = sess.run(
            spectrogram,
            feed_dict={
                wav_data_placeholder: np.reshape(wav_data, (-1, 1))})
    
    return spectrogram


def get_mfcc(filename, input_width, window_size_samples, window_stride_samples, sess, sample_rate=16000):
    """Saves audio sample data to a .wav audio file.

    Args:
        filename: Path to save the file to.
        sess: current session being run.
    Returns:
        2D Numpy array holding the MFCC data as floats.
    """
    #       The number of buckets in each FFT row of the spectrogram is calculated
    #       by the number of input samples in each frame. This can be very high, 
    #       with a 160 sample window, for example, yielding 127 buckets. 
    #       We don't need this amount of information for classification, 
    #       so we often try to minimize it to achieve a smaller outcome. 
    #       That is what the MFCC algorithm does.
    spectrogram = get_spectrogram(filename, window_size_samples, window_stride_samples, sess)
    spectrogram_placeholder = tf.compat.v1.placeholder(tf.float32, [None]+list(spectrogram.shape)[1:])
    mfcc = audio_ops.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count= input_width)
    tf.compat.v1.summary.image(
            'mfcc', tf.expand_dims(mfcc, -1), max_outputs=1)
    
    mfcc = sess.run(
            mfcc,
            feed_dict={
                spectrogram_placeholder: spectrogram})
    return mfcc



def get_label(file_path):
    """ Return the Label on each file path using the Parent's Directory Name
    """
    parts = file_path.split(os.path.sep)
    return parts[-2]



def run_mfcc(filename, input_width, window_size_samples, window_stride_samples, sess, sample_rate=16000):
        """ Run MFCC on a .wav file
        
        Args:
            filename: the path to wav_file
            sess: Current Session being run
        Returns:
            Return 1-D of mfcc with 1960 data points

        """
        wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
        wav_loader = io_ops.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=sample_rate)
        #background_clamp = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
        spectrogram = audio_ops.audio_spectrogram(
              wav_decoder.audio,
              window_size=window_size_samples,
              stride=window_stride_samples,
              magnitude_squared=True)

        mfcc = audio_ops.mfcc(
                spectrogram,
                wav_decoder.sample_rate,
                dct_coefficient_count= input_width)
        return sess.run(
                mfcc,
                feed_dict={
                    wav_filename_placeholder: filename}).flatten()
    
