import time
import os
#import neccesary packages
import tensorflow_hub as hub
import tensorflow as tf
import pickle
from keras import backend as K
import numpy as np
from sklearn_extra.cluster import KMedoids
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from datetime import datetime
from scipy.spatial import distance_matrix
import sys

from transformers import BertTokenizer, TFBertModel

import keras.backend as K
import operator

from tensorflow.keras import layers, Model, regularizers


def make_variables(tf_name, k1, k2, initializer):
    return tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32), trainable=True, name=tf_name)


# prototype layer
class prototypeLayer_Bert(keras.layers.Layer):
    def __init__(self, k_protos, vect_size, k_cents):
        super(prototypeLayer_Bert, self).__init__(name='proto_layer')
        self.n_protos = k_protos
        self.vect_size = vect_size
        self.prototypes = make_variables("prototypes", k_protos, vect_size,
                                         initializer=tf.constant_initializer(k_cents))

    @tf.function
    def call(self, inputs):
        tmp1 = tf.expand_dims(inputs, 2)

        tmp1 = tf.broadcast_to(tmp1, [tf.shape(tmp1)[0], tf.shape(tmp1)[1], self.n_protos, self.vect_size])
        tmp2 = tf.broadcast_to(self.prototypes,
                               [tf.shape(tmp1)[0], tf.shape(tmp1)[1], self.n_protos, self.vect_size])
        tmp3 = tmp1 - tmp2
        tmp4 = tmp3 * tmp3
        distances = tf.reduce_sum(tmp4, axis=3)

        return distances, self.prototypes


# distance layer: to convert the full distance matrix to sparse similarity matrix
class distanceLayer(keras.layers.Layer):
    def __init__(self):
        super(distanceLayer, self).__init__(name='distance_layer')
        self.a = 0.1
        self.beta = 1e6

    def e_func(self, x, e=2.7182818284590452353602874713527):
        return tf.math.pow(e, -(self.a * x))

    @tf.function
    def call(self, full_distances):
        min_dist_ind = tf.nn.softmax(-full_distances * self.beta)
        e_dist = self.e_func(full_distances) + 1e-8
        dist_hot_vect = min_dist_ind * e_dist
        return dist_hot_vect


class PrototypeCNN_Bert(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size, tokenizer, bert_model, word_idx_map, \
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):

        super(PrototypeCNN, self).__init__()
        self.k_protos = k_protos
        self.vect_size = vect_size
        self.full_distences = None
        self.full_onehot_distances = None
        self.vocab_size = vocab_size
        # self.tokenizer = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_size,weights = [pretrained_embeddings],trainable=True)  # Optional: Set to True if you want to fine-tune the embeddings during training
        self.tokenizer = tokenizer
        self.embedding = bert_model

        self.convs = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.Sequential([
                layers.Conv2D(num_filters, (filter_size, embedding_size),
                              padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1),
                                    strides=(1, 1), padding='valid')])
            self.convs.append(conv_block)

        self.flatten = layers.Flatten()
        self.distance_layer = distanceLayer()
        self.dropout = layers.Dropout(dropout_keep_prob)  # keep_prob will be supplied by call argument
        self.fc = layers.Dense(num_classes,
                               kernel_regularizer=regularizers.l2(l2_reg_lambda),
                               activation='softmax')

    def init_prototypelayer(self, k_cents):
        self.proto_layer = prototypeLayer(self.k_protos, self.vect_size, k_cents)

    def call(self, x):
        # Embedding layer
        x = self.embedding(self.tokenizer(x, padding=True, truncation=True, max_length=self.vocab_size))
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.convs:
            c = conv(x)
            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)

        x = tf.expand_dims(x, axis=0)
        full_distances, protos = self.proto_layer(x)

        dist_hot_vect = self.distance_layer(full_distances)

        x = self.dropout(dist_hot_vect)
        x = self.fc(x)
        x = tf.squeeze(x, axis=0)
        # return x, self.fc.weights[0], self.fc.weights[1]

        return x

    def embed(self, x):
        # Embedding layer
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.convs:
            # print(x.shape)

            c = conv(x)
            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)

        return x

    def full_distance(self, x):

        x = self.embedding(x)
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.convs:
            c = conv(x)
            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)

        x = tf.expand_dims(x, axis=0)
        full_distances, protos = self.proto_layer(x)

        return full_distances

    def one_hot_distance(self, x):

        # Embedding layer
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.convs:
            c = conv(x)
            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)

        x = tf.expand_dims(x, axis=0)
        full_distances, protos = self.proto_layer(x)

        dist_hot_vect = self.distance_layer(full_distances)

        return dist_hot_vect
