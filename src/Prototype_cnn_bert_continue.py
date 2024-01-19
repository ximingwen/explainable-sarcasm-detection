#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import os
# import neccesary packages
# import tensorflow_hub as hub
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

# In[54]:


from transformers import BertTokenizer, TFBertModel

# In[55]:


import keras.backend as K
import operator

# In[56]:


from tensorflow.keras import layers, Model, regularizers


def make_variables(tf_name, k1, k2, initializer):
    return tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32), trainable=True, name=tf_name)


# prototype layer
class prototypeLayer(keras.layers.Layer):
    def __init__(self, k_protos, vect_size, k_cents):
        super(prototypeLayer, self).__init__(name='proto_layer')
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

    def __init__(self, sequence_length, num_classes, tokenizer, bert_model, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):

        super(PrototypeCNN_Bert, self).__init__()
        self.k_protos = k_protos
        self.vect_size = vect_size
        self.full_distences = None
        self.full_onehot_distances = None
        self.embedding = bert_model
        self.tokenizer = tokenizer
        self.max_l = sequence_length

        RNN_CELL_SIZE = 128
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
        self.LSTM = LSTM(RNN_CELL_SIZE, return_sequences=True, return_state=True)
        self.dropout = layers.Dropout(dropout_keep_prob)  # keep_prob will be supplied by call argument
        self.fc = layers.Dense(num_classes,
                               kernel_regularizer=regularizers.l2(l2_reg_lambda),
                               activation='softmax')

    def init_prototypelayer(self, k_cents):
        self.proto_layer = prototypeLayer(self.k_protos, self.vect_size, k_cents)

    def call(self, x):

        # Embedding layer
        x = self.tokenizer(x, padding="max_length", max_length=self.max_l, return_tensors="tf", truncation=True)
        x = self.embedding(input_ids=x["input_ids"], attention_mask=x["attention_mask"], output_hidden_states=True)[0]
        x = tf.expand_dims(x, -1)

        # batch * 768
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

        # 1*batch_size*10

        #         lstmop, forward_h, forward_c = self.LSTM(tf.squeeze(dist_hot_vect))
        #         z1 = self.fc(lstmop[:, -1, :])
        #         z = tf.squeeze(z1, axis=0)

        x = self.dropout(dist_hot_vect)
        x = self.fc(x)
        x = tf.squeeze(x, axis=0)

        # return x, self.fc.weights[0], self.fc.weights[1]

        return x

    def embed(self, x):
        # Embedding layer

        x = self.tokenizer(x, padding="max_length", max_length=self.max_l, return_tensors="tf", truncation=True)
        x = self.embedding(input_ids=x["input_ids"], attention_mask=x["attention_mask"])[0]
        x = tf.expand_dims(x, -1)  # 2*200*768*

        pooled_outputs = []
        for conv in self.convs:
            c = conv(x)

            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)

        return x

    def full_distance(self, x):

        x = tokenizer(x, padding="max_length", max_length=200, return_tensors="tf")
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)  # 2*200*768*1

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
        return pooled_outputs

        # # Combine all the pooled features
        # x = tf.concat(pooled_outputs, axis=-1)
        # x = self.flatten(x)

        # x = tf.expand_dims(x, axis=0)
        # full_distances, protos = self.proto_layer(x)

        # dist_hot_vect = self.distance_layer(full_distances)

        # return dist_hot_vect


# In[57]:


# this method simple project prototypes to the closest sentences in
# sample_sent_vects
def projection(sample_sentences, sample_sent_vects, data_size=10000):
    prototypes = ProtoCNN.proto_layer.prototypes
    d_pos = {}
    # for each prototype
    for p_count, p in enumerate(prototypes):
        print('[db] p_count = ', p_count)
        s_count = 0
        d_pos[p_count] = {}
        # find its distances to all sample sentences
        for i, s in enumerate(sample_sent_vects[:data_size]):
            if len(sample_sentences[i]) < 5 or len(sample_sentences[i]) > 100:
                continue
            d_pos[p_count][i] = np.linalg.norm(sample_sent_vects[i] - p)
            s_count += 1
    # sort those distances, then assign the closest ones to new prototypes
    new_protos = []
    for p_count, p in enumerate(prototypes):
        sorted_d = sorted(d_pos[p_count].items(), key=operator.itemgetter(1))
        new_protos.append(sample_sent_vects[sorted_d[0][0]])
    # return these values

    return new_protos


# In[58]:


# show the list of prototypes
def showPrototypes(sample_sentences, sample_sent_vects, sample_y, k_protos=10, printOutput=False, k_closest_sents=20):
    prototypes = ProtoCNN.proto_layer.prototypes.numpy()
    # data_size = 10000
    d_pos = {}
    data_size = 150000
    for p_count, p in enumerate(prototypes):

        s_count = 0
        d_pos[p_count] = {}
        for i, s in enumerate(sample_sent_vect[:data_size]):
            # if len(sample_sentences[i]) < 20 or len(sample_sentences[i]) > 100:
            if len(sample_sentences[i]) < 30 or sample_y[i][1] == 0:
                continue
            d_pos[p_count][i] = np.linalg.norm(sample_sent_vect[i] - p)
            s_count += 1

    mappedPrototypes = {}

    recorded_protos_score = {}
    print("Prototypes: ")
    for l in range(k_protos):
        # print("prototype index = ", l)
        recorded_protos_score[l] = {}
        sorted_d = sorted(d_pos[l].items(), key=operator.itemgetter(1))
        print(l)
        mappedPrototypes[l] = []
        for k in range(k_closest_sents):
            i = sorted_d[k][0]
            score = sorted_d[k][1]
            # print("[db] sorted_d ",sorted_d[0])
            # print("[db] sample_sentences[sorted_d[0][0]]: ",sample_sentences[sorted_d[0][0]])
            mappedPrototypes[l].append((sample_sentences[i].strip(), score, sample_y[i][1]))
            if k < 10:
                print(sorted_d[k], sample_sentences[i], sample_y[i][1])
        # print(mappedPrototypes[l])

    return mappedPrototypes


# In[59]:


# method to generate the number of closest sentences to each prototype
def protoFreq(self, sample_sent_vect):
    d = {}
    for sent in sample_sent_vect:
        sent_dist = {}
        for i, p in enumerate(self.prototypes):
            sent_dist[i] = np.linalg.norm(sent - p)
            if i not in d:
                d[i] = 0
        sorted_sent_d = sorted(sent_dist.items(), key=operator.itemgetter(1))
        # print(sorted_sent_d)
        picked_protos = sorted_sent_d[0][0]
        d[picked_protos] += 1
    print("Prototype freq = ", d)
    x = sorted(d.items(), key=lambda item: item[1], reverse=True)
    print("sorted :", x)


# re-train the model with new pruned prototype


# In[60]:


def pruningTrain(self, new_k_protos, x_train, y_train, x_test, y_test):
    # print("[db] self prototypes: ",self.prototypes)
    k_cents = self.prototypes[:new_k_protos]
    k_cents = [p.numpy() for p in k_cents]
    # print("[db] k_cents = ",k_cents)
    self.createModel(k_cents=k_cents, k_protos=new_k_protos)
    self.train(x_train, y_train, x_test, y_test)


# generate the sentence value for each prototype
# and 10 closest sentences to it


# In[61]:


def showTrajectory(self, input, sample_sentences, sample_vect):
    if len(self.mappedPrototypes) == 0:
        self.showPrototypes(sample_sentences, sample_vect, printOutput=False)
    prototypes = [self.mappedPrototypes[k].strip() for k in self.mappedPrototypes]
    vP, vS = self.embed(prototypes), self.embed(input)
    dStoP = {}
    for sCount, s in enumerate(vS):
        dStoP[sCount] = {}
        for i, p in enumerate(vP):
            dStoP[sCount][i] = np.linalg.norm(vS[sCount] - p)

    mappedProtos, mappedScore, mappedDist = [], [], []
    for sCount, s in enumerate(vS):
        sorted_d = sorted(dStoP[sCount].items(), key=operator.itemgetter(1))
        mappedProtos.append(prototypes[sorted_d[0][0]])

    # for small dataset, we use a pretrained sentiment model. We can use any
    # model for sentiment scores
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid_obj = SentimentIntensityAnalyzer()
    print("[db] mappedProtos ", mappedProtos)
    scores = []
    for s in mappedProtos:
        # sentiment_dict = sid_obj.polarity_scores(s)
        scores.append(0.5 + sid_obj.polarity_scores(s)['compound'] / 2)
    return scores


# In[62]:


dev_sample_percentage = .1

# Model Hyperparameters
embedding_dim = 768
filter_sizes = "3,4,5"
num_filters = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.5
max_l = 100
# Training parameters
batch_size = 4096
num_epochs = 100
evaluate_every = 100
checkpoint_everyt = 100
num_checkpoints = 5

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# # Data preprocessing

# In[63]:


timestamp = str(int(time.time()))

out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("output directory: ", out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# Data Preparation
# ==================================================

# Load data

print("loading data...")
x = pickle.load(open("./mainbalancedpickle.p", "rb"))
revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
print("data loaded!")  # Load data

# In[64]:


revs[0]

# In[65]:


bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)

# In[16]:


max_l = 100

x_text = []
y = []

test_x = []
test_y = []

for i in range(len(revs)):
    if revs[i]['split'] == 1:
        x_text.append(revs[i]['text'])
        y.append(revs[i]['label'])
    else:
        test_x.append(revs[i]['text'])
        test_y.append(revs[i]['label'])

y = np.asarray(y)
y_test = np.asarray(test_y)

# In[17]:


x_text[:5]

# In[18]:


shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = np.asarray(x_text)[shuffle_indices]
y_shuffled = np.asarray(y)[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation

dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

x_train = np.asarray(x_train)
x_dev = np.asarray(x_dev)
y_train = np.asarray(y_train)
y_dev = np.asarray(y_dev)

# In[19]:


x_train[0], y_train[0]

# In[20]:


for layer in bert_model.layers:
    layer.trainable = False

# In[21]:


word_idx_map["@"] = 0
rev_dict = {v: k for k, v in word_idx_map.items()}

# In[22]:


k_protos, vect_size = 10, 384

# In[23]:


ProtoCNN = PrototypeCNN_Bert(sequence_length=max_l,
                             num_classes=len(y_train[0]),
                             tokenizer=tokenizer,
                             bert_model=bert_model,
                             embedding_size=embedding_dim,
                             filter_sizes=list(map(int, filter_sizes.split(","))),
                             num_filters=num_filters,
                             l2_reg_lambda=l2_reg_lambda,
                             dropout_keep_prob=dropout_keep_prob,
                             k_protos=k_protos,
                             vect_size=vect_size)

# In[24]:


data = x_text[:2]
y = ProtoCNN.embed(data)

# In[25]:


import random
import copy

# In[29]:


# random.shuffle(x_text)
sample_sentences = x_text[:15000]
sample_sentences_vects = []
for i in range(300):
    batch = sample_sentences[i * 50:(i + 1) * 50]
    vect = ProtoCNN.embed(batch)
    sample_sentences_vects.append(vect.numpy())

# In[30]:


sample_sentences_vect = np.concatenate(sample_sentences_vects, axis=0)

k_protos = 10
kmedoids = KMedoids(n_clusters=k_protos, random_state=0).fit(sample_sentences_vect)
k_cents = kmedoids.cluster_centers_

# In[33]:


ProtoCNN.init_prototypelayer(k_cents)

y = ProtoCNN(x_text[:2])

batch_size = 64
accumulated_steps = 70

# In[37]:


# timestamp = str(int(time.time()))
# Output directory for models and summaries
# out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
# print("Writing to {}\n".format(out_dir))

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
# checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
#
# checkpoint_prefix = os.path.join(checkpoint_dir, "model")


# ProtoCNN = tf.keras.models.load_model(os.path.join(out_dir,"my_weights-finetune.pt"))


# In[41]:


# We use Adam optimizer with default learning rate 0.0001.
# Change this value based on your preference
out_dir = "/big/xw384/schoolwork/NLP+DEEP LEARNING/Project/CASCADE/src/runs/10_31"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
opt = tf.keras.optimizers.Adam(learning_rate=.0001)
# ProtoCNN.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
criterion = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.SUM)


# In[42]:


# loaded_object = pickle.load(open(os.path.join(out_dir,"optimizer.pt"), 'rb'))
# ProtoCNN.optimizer.set_weights(loaded_object)


# In[43]:


# i = 0

# maxEvalRes = 0

# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath = checkpoint_dir,  # Specify the path to save the checkpoints
#     save_weights_only=True,  # Save only the model weights
#     monitor='val_loss',  # Monitor the validation loss for saving the best weights
#     save_best_only=True,  # Save only the best weights based on the monitored metric
#     verbose=1  # Print a message when a checkpoint is saved
# )


# In[44]:


class DataLoader:
    def __init__(self, data, labels, batch_size=200, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(data))

    def __len__(self):
        # Returns the number of batches
        return int(np.ceil(len(self.data) / self.batch_size))

    def __iter__(self):
        # Shuffles the indexes if required
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Yield batches
        for i, start_idx in enumerate(range(0, len(self.data), self.batch_size)):
            end_idx = min(start_idx + self.batch_size, len(self.data))
            batch_indexes = self.indexes[start_idx:end_idx]
            yield i, self.data[batch_indexes].tolist(), self.labels[batch_indexes]


# In[45]:


batch_size = 64
accumulation_steps = 70

train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
valid_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

# In[46]:


train_loader = DataLoader(x_train, y_train, batch_size=64)

# In[47]:


valid_loader = DataLoader(x_dev, y_dev, batch_size=64)

# In[48]:


EPOCHS = 4000

# In[49]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In[50]:
f = open("loss_log.txt", "w")
train_losses = []
valid_losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    accumulated_gradients = [tf.zeros_like(var) for var in ProtoCNN.trainable_weights]
    accumulated_loss = 0
    best_loss_so_far = float("inf")
    train_accuracy_metric.reset_state()
    for i, inputs, labels in train_loader:

        with tf.GradientTape() as tape:

            predictions = ProtoCNN(inputs, training=True)
            loss = criterion(labels, predictions)
            gradients = tape.gradient(loss, ProtoCNN.trainable_weights)

            # Accumulate gradients
            accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]
            accumulated_loss += loss
            train_accuracy_metric.update_state(labels, predictions)

            # Apply gradients every accumulation_steps or at the last batch
        if (i + 1) % accumulation_steps == 0 or i == len(train_loader) - 1:
            accumulated_gradients = [grad / accumulation_steps for grad in accumulated_gradients]
            opt.apply_gradients(zip(accumulated_gradients, ProtoCNN.trainable_weights))
            accumulated_gradients = [tf.zeros_like(var) for var in ProtoCNN.trainable_variables]

            print(
                f"Epoch: {epoch + 1}, Loss: {accumulated_loss.numpy() / (batch_size * accumulation_steps)} {batch_size * (i + 1)}/139232 train accuracy:{train_accuracy_metric.result().numpy()}")

            accumulated_loss = 0
        epoch_loss += loss

    f.write(
        f"Epoch: {epoch + 1}, epoch Loss: {epoch_loss.numpy()}  train accuracy:{train_accuracy_metric.result().numpy()}")

    valid_loss = 0
    y_true = None
    y_pred = None
    valid_accuracy_metric.reset_state()
    for i, inputs, labels in valid_loader:

        with tf.GradientTape() as tape:

            predictions = ProtoCNN(inputs, training=False)
            loss = criterion(labels, predictions)
            accumulated_loss += loss
            valid_loss += loss
            valid_accuracy_metric.update_state(labels, predictions)

        if (i + 1) % accumulation_steps == 0 or i == len(valid_loader) - 1:
            print(
                f"Epoch: {epoch + 1}, Loss: {accumulated_loss.numpy() / (batch_size * accumulation_steps)} {batch_size * (i + 1)}/15488 valid accuracy:{valid_accuracy_metric.result().numpy()}")

            accumulated_loss = 0

    if valid_loss < best_loss_so_far:
        print("find better loss")
        ProtoCNN.save_weights(os.path.join(out_dir, "best_classifier.ckpt"))
        pickle.dump(opt.get_weights(), open(os.path.join(out_dir, "optimizer.pt"), "wb+"))
        best_loss_so_far = valid_loss

    f.write(
        f"Epoch: {epoch + 1}, valid Loss: {valid_loss.numpy()}  valid accuracy:{valid_accuracy_metric.result().numpy()}")
    train_losses.append(epoch_loss)
    valid_losses.append(valid_loss)

f.close()
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Optional: set figure size
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Save the plot as an image
plt.savefig('training_validation_loss.png')

# Optionally, display the plot
plt.show()

with open('losses.pkl', 'wb') as file:
    pickle.dump((train_losses, valid_losses), file)
