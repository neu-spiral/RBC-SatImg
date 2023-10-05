''' This script defines custom metrics and loss functions.

The Adaptive Max-Pool Loss acts as a weighting function that multiplies a
loss value with the maximum loss values within an NxN neighborhood.
An earlier version of this loss function described in:

F. Isikdogan, A.C. Bovik, and P. Passalacqua,
"Learning a River Network Extractor using an Adaptive Loss Function,"
IEEE Geoscience and Remote Sensing Letters, 2018.
'''

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def running_recall(y_true, likelihood):
    TP = K.sum(K.round(K.clip(y_true * likelihood, 0, 1)))
    TP_FN = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = TP / (TP_FN + K.epsilon())
    return recall

def running_precision(y_true, likelihood):
    TP = K.sum(K.round(K.clip(y_true * likelihood, 0, 1)))
    TP_FP = K.sum(K.round(K.clip(likelihood, 0, 1)))
    precision = TP / (TP_FP + K.epsilon())
    return precision

def running_f1(y_true, likelihood):
    precision = running_precision(y_true, likelihood)
    recall = running_recall(y_true, likelihood)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def adaptive_maxpool_loss(y_true, likelihood, alpha=0.25):
    likelihood = K.clip(likelihood, K.epsilon(), 1. - K.epsilon())
    positive = -y_true * K.log(likelihood) * alpha
    negative = -(1. - y_true) * K.log(1. - likelihood) * (1-alpha)
    pointwise_loss = positive + negative
    max_loss = tf.keras.layers.MaxPool2D(pool_size=8, strides=1, padding='same')(pointwise_loss)
    x = pointwise_loss * max_loss
    x = K.mean(x, axis=-1)
    return x