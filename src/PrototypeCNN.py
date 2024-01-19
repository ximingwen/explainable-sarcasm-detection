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


from tensorflow.keras import layers, Model, regularizers

def make_variables(tf_name, k1, k2, initializer):
     
    return tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32), trainable=True, name=tf_name)

#prototype layer
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

#distance layer: to convert the full distance matrix to sparse similarity matrix
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
    
    
class PrototypeCNN(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, pretrained_embeddings, word_idx_map,\
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):

        
        super(PrototypeCNN, self).__init__()
        self.k_protos = k_protos
        self.vect_size = vect_size
       
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_size,weights = [pretrained_embeddings],trainable=True)  # Optional: Set to True if you want to fine-tune the embeddings during training

        
        self.convs = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.Sequential([
                layers.Conv2D(num_filters, (filter_size, embedding_size), 
                              padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1), 
                                    strides=(1,1), padding='valid')])
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

        x = self.dropout(dist_hot_vect)
        x = self.fc(x)
        x = tf.squeeze(x, axis=0)
        #return x, self.fc.weights[0], self.fc.weights[1]
        return x
    
    def embed(self,x):
        # Embedding layer
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.convs:
            #print(x.shape)
            
            c = conv(x)
            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)
        
        return x
    
   
 class PrototypeCNN_Bert(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, pretrained_embeddings, word_idx_map,\
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):

        
        super(PrototypeCNN_Bert, self).__init__()
        self.k_protos = k_protos
        self.vect_size = vect_size
       
        bert_model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = TFBertModel.from_pretrained(bert_model_name)
        
        self.convs = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.Sequential([
                layers.Conv2D(num_filters, (filter_size, embedding_size), 
                              padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1), 
                                    strides=(1,1), padding='valid')])
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

        x = self.dropout(dist_hot_vect)
        x = self.fc(x)
        x = tf.squeeze(x, axis=0)
        #return x, self.fc.weights[0], self.fc.weights[1]
        return x
    
    def embed(self,x):
        # Embedding layer
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.convs:
            #print(x.shape)
            
            c = conv(x)
            pooled_outputs.append(c)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, axis=-1)
        x = self.flatten(x)
        
        return x

if __name__ == "__main__":


    dev_sample_percentage = .1


    # Model Hyperparameters
    embedding_dim = 300
    filter_sizes ="3,4,5"
    num_filters = 128
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.5
    max_l = 100

    # Training parameters
    batch_size = 4096
    num_epochs = 1000
    evaluate_every = 100
    checkpoint_everyt = 100
    num_checkpoints = 5

    # Misc Parameters
    allow_soft_placement = True
    log_device_placement = False
    
    timestamp = str(int(time.time()))

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("output directory: ", out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Data Preparation
    # ==================================================

    # Load data

    print("loading data...")
    x = pickle.load(open("./mainbalancedpickle.p","rb"))
    revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")# Load data

    max_l = 100

    x_text = []
    y = []

    test_x = []
    test_y = []

    for i in range(len(revs)):
        if revs[i]['split']==1:
            x_text.append(revs[i]['text'])
            temp_y = revs[i]['label']
            y.append(temp_y)
        else:
            test_x.append(revs[i]['text'])
            test_y.append(revs[i]['label'])  

    y = np.asarray(y)
    test_y = np.asarray(test_y)

    # get word indices
    x = []
    for i in range(len(x_text)):
        x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))

    x_test = []
    for i in range(len(test_x)):
        x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))

    # padding
    for i in range(len(x)):
        if( len(x[i]) < max_l ):
            x[i] = np.append(x[i],np.zeros(max_l-len(x[i])))
        elif( len(x[i]) > max_l ):
            x[i] = x[i][0:max_l]
    x = np.asarray(x)

    for i in range(len(x_test)):
        if( len(x_test[i]) < max_l ):
            x_test[i] = np.append(x_test[i],np.zeros(max_l-len(x_test[i])))        
        elif( len(x_test[i]) > max_l ):
            x_test[i] = x_test[i][0:max_l]
    x_test = np.asarray(x_test)
    y_test = test_y

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]


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
    word_idx_map["@"] = 0
    rev_dict = {v: k for k, v in word_idx_map.items()}



    k_protos, vect_size = 10, 384
    
    ProtoCNN = PrototypeCNN(sequence_length=max_l,
    num_classes=len(y_train[0]),
    vocab_size=len(W),
    pretrained_embeddings = W,
    word_idx_map = word_idx_map,
    embedding_size=embedding_dim,
    filter_sizes=list(map(int, filter_sizes.split(","))),
    num_filters=num_filters,
    l2_reg_lambda=l2_reg_lambda,
    dropout_keep_prob = dropout_keep_prob,
    k_protos = k_protos,
    vect_size = vect_size)
    
    
    #choose with data to sample
    sample_sentences = x_train[:5000]
    #compute vector values of sentences
    sample_sent_vect = ProtoCNN.embed(sample_sentences)
    
    
    
    k_protos = 10
    kmedoids = KMedoids(n_clusters=k_protos, random_state=0).fit(sample_sent_vect)
    k_cents = kmedoids.cluster_centers_
    print(k_cents.shape)
    
    ProtoCNN.init_prototypelayer(k_cents)
    
    
    timestamp = str(int(time.time()))
    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    
    
    #We use Adam optimizer with default learning rate 0.0001.
    #Change this value based on your preference
    opt = tf.keras.optimizers.Adam(learning_rate=.0001)
    ProtoCNN.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])



    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_dir,  # Specify the path to save the checkpoints
        save_weights_only=True,  # Save only the model weights
        monitor='val_loss',  # Monitor the validation loss for saving the best weights
        save_best_only=True,  # Save only the best weights based on the monitored metric
        verbose=1  # Print a message when a checkpoint is saved
    )    
    ProtoCNN.fit(x_train,y_train, batch_size = 4096, epochs=num_epochs, verbose=1, validation_data= (x_dev, y_dev))


