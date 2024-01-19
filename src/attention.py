#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
import os
import csv
from time import sleep
import pickle, argparse
from sklearn_extra.cluster import KMedoids
from tensorflow.keras import layers, Model, regularizers
from tensorflow import keras 
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm
import pandas as pd


def make_variables(tf_name, k1, k2, initializer):
     
    return tf.Variable(initializer(shape=[k1, k2], dtype=tf.float32), trainable=True, name=tf_name)


def pw_distance(A):
    r = tf.reduce_sum(A * A, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return D

def tight_pos_sigmoid_offset(x, offset, e=2.7182818284590452353602874713527):
    return 1 / (1 + tf.math.pow(e, (1 * (offset * x - 0.8))))


class DataLoader:
    def __init__(self, data, batch_size=200, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
    

    def __len__(self):
        # Returns the number of batches
        return int(np.ceil(len(self.data) / self.batch_size))

    def __iter__(self):
        # Shuffles the indexes if required
        data = pd.DataFrame(self.data).to_numpy()
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/self.batch_size) + 1
      
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)
            output = list(zip(*shuffled_data[start_index:end_index]))
            yield output[0],  output[1],  output[2],  output[3]
       

            
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

    # @tf.function
    # def call(self, full_distances):
    #     min_dist_ind = tf.nn.softmax(-full_distances * self.beta)
    #     e_dist = self.e_func(full_distances) + 1e-8
    #     dist_hot_vect = min_dist_ind * e_dist
    #     return dist_hot_vect

    @tf.function
    def call(self, full_distances):
        e_dist = self.e_func(full_distances) + 1e-8
        dist_hot_vect = tf.squeeze(e_dist, axis=0)
        return dist_hot_vect


class TextCNN(tf.keras.Model):
    """
    A CNN for text classification in TensorFlow 2.x.
    Uses an embedding layer, followed by a convolutional, max-pooling, and softmax layer.
    """
    
    def __init__(
        self, sequence_length, num_classes, tokenizer, bert_model, user_embeddings, topic_embeddings, embedding_size, filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):
        super(TextCNN, self).__init__()
        self.max_l = sequence_length
        self.l2_reg_lambda = l2_reg_lambda
        l2_regularizer = tf.keras.regularizers.l2(l2_reg_lambda)
        # Embedding layer
        self.tokenizer = tokenizer
        self.embedding = bert_model
        self.num_filters = num_filters
        self.filters_sizes = filter_sizes
        self.k_protos = k_protos
        self.vect_size = vect_size
        
        self.user_embedding = tf.keras.layers.Embedding(input_dim=user_embeddings.shape[0], output_dim=user_embeddings.shape[1], weights=[user_embeddings], trainable=False)
        self.topic_embedding = tf.keras.layers.Embedding(input_dim=topic_embeddings.shape[0], output_dim=topic_embeddings.shape[1], weights=[topic_embeddings], trainable=False)
        self.distance_layer = distanceLayer()

        self.conv_layers = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.Sequential([
                layers.Conv2D(num_filters, (filter_size, embedding_size), 
                              padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1), 
                                    strides=(1,1), padding='valid')])
            self.conv_layers.append(conv_block)
        self.concat_layer = tf.keras.layers.Concatenate()
        #self.last_dense = tf.keras.layers.Dense(100, activation='relu')
        self.user_topic_dense = tf.keras.layers.Dense(400, activation='relu')
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep_prob)
        # Final dense layer with L2 regularization
        self.final_dense = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2_regularizer)

    def init_prototypelayer(self, res_cents, user_cents):
        
        self.response_proto_layer = prototypeLayer(self.k_protos, self.vect_size, res_cents)
        self.user_proto_layer = prototypeLayer(self.k_protos, 100, user_cents)
       

    def call(self, inputs):
       

        input_content, input_author, input_topic = inputs

       
         # Embedding layer
        x = self.tokenizer(input_content, padding = "max_length", max_length=self.max_l, return_tensors ="tf",truncation = True )
        x = self.embedding(input_ids = x["input_ids"], attention_mask = x["attention_mask"], output_hidden_states =True)[0]
        x = tf.expand_dims(x, -1)


        pooled_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            pooled_outputs.append(conv_out)

        num_filters_total = self.num_filters * len(self.filters_sizes)
        h_pool = tf.concat(pooled_outputs, axis=1)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        
        x = tf.expand_dims(h_pool_flat, axis=0)
        full_distances, res_protos = self.response_proto_layer(x)
        res_vect = self.distance_layer(full_distances)

        user_embeddings = self.user_embedding(input_author)

      
        full_distances, user_protos = self.user_proto_layer(tf.expand_dims(user_embeddings, axis=0))
        user_vect = self.distance_layer(full_distances)


        topic_embeddings = self.topic_embedding(input_topic)
        
        combined_vectors = self.concat_layer([res_vect, user_vect, topic_embeddings])
        combined_vector_final = self.user_topic_dense(combined_vectors)

        
        combined_vector_final = self.dropout(combined_vector_final)

        scores = self.final_dense(combined_vector_final)

        return scores, res_protos, user_protos
    
    def embed_res(self, x):
         # Embedding layer
        x = self.tokenizer(x, padding = "max_length", max_length=self.max_l, return_tensors ="tf",truncation = True )
        x = self.embedding(input_ids = x["input_ids"], attention_mask = x["attention_mask"], output_hidden_states =True)[0]
        x = tf.expand_dims(x, -1)


        
        pooled_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            pooled_outputs.append(conv_out)

           
        
        num_filters_total = self.num_filters * len(self.filters_sizes)
        
        h_pool = tf.concat(pooled_outputs, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    
        return h_pool_flat

    def embed_user(self, x):

        user_embeddings = self.user_embedding(x)

        return user_embeddings





 

#####################  GPU Configs  #################################

# Selecting the GPU to work on
if __name__ == "__main__":
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    
    np.random.seed(10)

    
    # Selecting the GPU to work on
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # Desired graphics card config
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    
    # Data loading params
    parser.add_argument("--dev_sample_percentage", type=float, default=0.1,
                        help="Percentage of the training data to use for validation")
    
    # Model Hyperparameters
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Dimensionality of character embedding (default: 128)")
    parser.add_argument("--filter_sizes", type=str, default="3,4,5",
                        help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument("--num_filters", type=int, default=128,
                        help="Number of filters per filter size (default: 128)")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5,
                        help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", type=float, default=0.5,
                        help="L2 regularization lambda (default: 0.0)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=60,
                        help="Batch Size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=4000,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--evaluate_every", type=int, default=100,
                        help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save model after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="Number of checkpoints to store (default: 5)")
    
    # Misc Parameters
    parser.add_argument("--allow_soft_placement", action="store_true",
                        help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", action="store_true",
                        help="Log placement of ops on devices")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    print("loading data...")
    x = pickle.load(open("./mainbalancedpickle.p","rb"))
    revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    print("data loaded!")# Load data
    
    print('loading wgcca embeddings...')
    wgcca_embeddings = np.load('./../users/user_embeddings/user_gcca_embeddings.npz')
    print('wgcca embeddings loaded')
    
    
    ids = np.concatenate((np.array(["unknown"]), wgcca_embeddings['ids']), axis=0)
    user_embeddings = wgcca_embeddings['G']
    unknown_vector = np.random.normal(size=(1,100))
    user_embeddings = np.concatenate((unknown_vector, user_embeddings), axis=0)
    user_embeddings = user_embeddings.astype(dtype='float32')
    
    wgcca_dict = {}
    for i in range(len(ids)):
        wgcca_dict[ids[i]] = int(i)
    
    csv_reader = csv.reader(open("./../discourse/discourse_features/discourse.csv"))
    topic_embeddings = []
    topic_ids = []
    for line in csv_reader:
        topic_ids.append(line[0])
        topic_embeddings.append(line[1:])
    topic_embeddings = np.asarray(topic_embeddings)
    topic_embeddings_size = len(topic_embeddings[0])
    topic_embeddings = topic_embeddings.astype(dtype='float32')
    print("topic emb size: ",topic_embeddings_size)
    
    topics_dict = {}
    for i in range(len(topic_ids)):
        try:
            topics_dict[topic_ids[i]] = int(i)
        except TypeError:
            print(i)
    
    max_l = 100
    
    x_text = []
    author_text_id = []
    topic_text_id = []
    y = []
    
    test_x = []
    test_topic = []
    test_author = []
    test_y = []
    
    for i in range(len(revs)):
        if revs[i]['split']==1:
            x_text.append(revs[i]['text'])
            try:
                author_text_id.append(wgcca_dict['"'+revs[i]['author']+'"'])
            except KeyError:
                author_text_id.append(0)
            try:
                topic_text_id.append(topics_dict['"'+revs[i]['topic']+'"'])
            except KeyError:
                topic_text_id.append(0)
            y.append(revs[i]['label'])
        else:
            test_x.append(revs[i]['text'])
            try:
                test_author.append(wgcca_dict['"'+revs[i]['author']+'"'])
            except:
                test_author.append(0)
            try:
                test_topic.append(topics_dict['"'+revs[i]['topic']+'"'])
            except:
                test_topic.append(0)
            test_y.append(revs[i]['label'])  
    
 
    y_test = test_y
    
    # # get word indices
    # x = []
    # for i in range(len(x_text)):
    # 	x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))
        
    # x_test = []
    # for i in range(len(test_x)):
    #     x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))
    
    # # padding
    # for i in range(len(x)):
    #     if( len(x[i]) < max_l ):
    #     	x[i] = np.append(x[i],np.zeros(max_l-len(x[i])))		
    #     elif( len(x[i]) > max_l ):
    #     	x[i] = x[i][0:max_l]
    # x = np.asarray(x)
    
    # for i in range(len(x_test)):
    #     if( len(x_test[i]) < max_l ):
    #         x_test[i] = np.append(x_test[i],np.zeros(max_l-len(x_test[i])))        
    #     elif( len(x_test[i]) > max_l ):
    #         x_test[i] = x_test[i][0:max_l]
    # x_test = np.asarray(x_test)
    
    topic_train = np.asarray(topic_text_id)
    topic_test = np.asarray(test_topic)
    author_train = np.asarray(author_text_id)
    author_test = np.asarray(test_author)
    
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = np.asarray(x_text)[shuffle_indices]
    y_shuffled = np.asarray(y)[shuffle_indices]


    topic_train_shuffled = topic_train[shuffle_indices]
    author_train_shuffled = author_train[shuffle_indices]
    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    
    dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    topic_train, topic_dev = topic_train_shuffled[:dev_sample_index], topic_train_shuffled[dev_sample_index:]
    author_train, author_dev = author_train_shuffled[:dev_sample_index], author_train_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    
    x_train = np.asarray(x_train)
    x_dev = np.asarray(x_dev)
    author_train = np.asarray(author_train)
    author_dev = np.asarray(author_dev)
    topic_train = np.asarray(topic_train)
    topic_dev = np.asarray(topic_dev)
    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    # word_idx_map["@"] = 0
    # rev_dict = {v: k for k, v in word_idx_map.items()}
    
    # Training
    # ==================================================

    bert_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = TFBertModel.from_pretrained(bert_model_name)

    for layer in bert_model.layers:
        layer.trainable = False

    k_protos, vect_size = 15, 384

    ProtoCNN = TextCNN(
        sequence_length=max_l,
        num_classes=len(y_train[0]) ,
        tokenizer = tokenizer,
        bert_model = bert_model,
        user_embeddings = user_embeddings,
        topic_embeddings = topic_embeddings,
        embedding_size=args.embedding_dim,
        filter_sizes=list(map(int, args.filter_sizes.split(","))),
        num_filters=args.num_filters,
        l2_reg_lambda=args.l2_reg_lambda,
        dropout_keep_prob = args.dropout_keep_prob,
        k_protos = k_protos,
        vect_size = vect_size)

    

 




    # random.shuffle(x_text)
    sample_sentences = x_text[:15000]
    sample_sentences_vects = []
    for i in range(300):
        batch = sample_sentences[i * 50:(i + 1) * 50]
        vect = ProtoCNN.embed_res(batch)
        sample_sentences_vects.append(vect.numpy())




    
    sample_sentences_vect = np.concatenate(sample_sentences_vects, axis=0)
    
   
    kmedoids = KMedoids(n_clusters=k_protos, random_state=0).fit(sample_sentences_vect)
    res_cents = kmedoids.cluster_centers_
    


    # random.shuffle(x_text)
    sample_users = author_text_id[:15000]
    sample_user_vects = []
    for i in range(300):
        batch = sample_users[i * 50:(i + 1) * 50]
        vect = ProtoCNN.embed_user(np.asarray(batch))
        sample_user_vects.append(vect.numpy())


    

    
    sample_users_vect = np.concatenate(sample_user_vects, axis=0)
    
   
    kmedoids = KMedoids(n_clusters=k_protos, random_state=0).fit(sample_users_vect)
    user_cents = kmedoids.cluster_centers_
   
    
    
    ProtoCNN.init_prototypelayer(res_cents, user_cents)


    predictions = ProtoCNN([x_train[:2].tolist(), author_train[:2], topic_train[:2]])

 
 

    # Define Training procedure
    
    opt = tf.keras.optimizers.Adam(learning_rate=.1e-3)
    criterion = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.SUM)

    train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    valid_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

    out_dir = "/big/xw384/schoolwork/NLP+DEEP LEARNING/Project/CASCADE/src/runs/two_prototype_layer-15_protos-add-loss/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)  

    
    
    # Generate batches

    train_loader = DataLoader(list(zip(x_train, author_train, topic_train, y_train)), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(list(zip(x_dev, author_dev, topic_dev, y_dev)), args.batch_size, shuffle = False)
    test_loader = DataLoader(list(zip(test_x, author_test, topic_test, y_test)), args.batch_size, shuffle = False)
    # Training loop. For each batch...

    accumulation_steps = 70
    

    train_loss = []
    train_acc = []
    dev_loss  = []
    dev_acc = []
    test_acc = []
    
    train_res_div_loss = []
    train_user_div_loss = []
    train_acc_loss = []

    dev_res_div_loss = []
    dev_user_div_loss = []
    dev_acc_loss = []
    best_loss_so_far = float("inf")
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_user_div_loss = 0
        epoch_res_div_loss = 0
        epoch_acc_loss = 0
        accumulated_gradients = [tf.zeros_like(var) for var in ProtoCNN.trainable_weights]
        accumulated_loss = 0
        
        train_accuracy_metric.reset_state()
        
        for i, inputs in tqdm(enumerate(train_loader)):
    
            x_batch, author_batch, topic_batch, y_batch = inputs     
    
            #x_batch = np.asarray(x_batch)
            author_batch = np.asarray(author_batch)
            topic_batch = np.asarray(topic_batch)
            y_batch = np.asarray(y_batch)
    
    
    
            with tf.GradientTape() as tape:
                predictions, res_protos, user_protos= ProtoCNN([x_batch, author_batch, topic_batch], training=True)
                acc_loss = criterion(y_batch, predictions)
                
                d = pw_distance(res_protos)
                diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
                diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
                d1 = d + diag_ones * tf.reduce_max(d)
                d2 = tf.reduce_min(d1, axis=1)
                min_d2_dist = tf.reduce_min(d2)
                # the third loss term
                res_div_loss = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8

                d = pw_distance(user_protos)
                diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
                diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
                d1 = d + diag_ones * tf.reduce_max(d)
                d2 = tf.reduce_min(d1, axis=1)
                min_d2_dist = tf.reduce_min(d2)
                # the third loss term
                user_div_loss = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8
                loss = acc_loss + 0.1*res_div_loss+0.1*user_div_loss
                gradients = tape.gradient(loss, ProtoCNN.trainable_weights)

                

              
                

    
            # Accumulate gradients
            accumulated_gradients = [acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)]
            accumulated_loss += loss
            train_accuracy_metric.update_state(y_batch, predictions)
    
                # Apply gradients every accumulation_steps or at the last batch
            if (i + 1) % accumulation_steps == 0 or i == len(train_loader)//args.batch_size - 1:
                accumulated_gradients = [grad / accumulation_steps for grad in accumulated_gradients]
                opt.apply_gradients(zip(accumulated_gradients, ProtoCNN.trainable_weights))
                accumulated_gradients = [tf.zeros_like(var) for var in ProtoCNN.trainable_variables]
    
                print(
                    f"Epoch: {epoch + 1}, Loss: {accumulated_loss.numpy() / (args.batch_size * accumulation_steps)} {args.batch_size * (i + 1)}/139232 train accuracy:{train_accuracy_metric.result().numpy()}")
    
                accumulated_loss = 0
           
            
            epoch_loss += loss
            epoch_res_div_loss += res_div_loss
            epoch_user_div_loss += user_div_loss
            epoch_acc_loss += acc_loss
            

         

           

        print(f"Epoch: {epoch+1}, epoch Loss: {epoch_loss }  train accuracy:{train_accuracy_metric.result().numpy()}\n")
        
        train_loss.append(epoch_loss )
        train_user_div_loss.append(epoch_user_div_loss)
        train_res_div_loss.append(epoch_res_div_loss)
        train_acc_loss.append(epoch_acc_loss)
        train_acc.append(train_accuracy_metric.result().numpy())
            
            
    
      
        
        # Divide into batches
        # dev_batches = data_helpers.batch_iter_dev(list(zip(x_dev, author_dev, topic_dev, y_dev)), args.batch_size)
        # dev_loss = []
        # ll = len(dev_batches)
        # conf_mat = np.zeros((2,2))

        valid_loss = 0
        valid_acc_loss = 0
        valid_user_div_loss = 0
        valid_res_div_loss = 0
        valid_accuracy_metric.reset_state()
        for x_batch, author_batch, topic_batch, y_batch in tqdm(dev_loader

            author_batch = np.asarray(author_batch)
            topic_batch = np.asarray(topic_batch)
            y_batch = np.asarray(y_batch)
    

          
            predictions, res_protos, user_protos= ProtoCNN([x_batch, author_batch, topic_batch], training=False)
            acc_loss = criterion(y_batch, predictions)
            
            d = pw_distance(res_protos)
            diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
            diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
            d1 = d + diag_ones * tf.reduce_max(d)
            d2 = tf.reduce_min(d1, axis=1)
            min_d2_dist = tf.reduce_min(d2)
            # the third loss term
            res_div_loss = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8

            d = pw_distance(user_protos)
            diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
            diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
            d1 = d + diag_ones * tf.reduce_max(d)
            d2 = tf.reduce_min(d1, axis=1)
            min_d2_dist = tf.reduce_min(d2)
            # the third loss term
            user_div_loss = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8


            loss = acc_loss + 0.1*res_div_loss+0.1*user_div_loss
          
           

        

           
            valid_loss += loss
            valid_res_div_loss += res_div_loss
            valid_user_div_loss += user_div_loss
            valid_acc_loss += acc_loss
            
            valid_accuracy_metric.update_state(y_batch, predictions)

        dev_loss.append(valid_loss)
        dev_user_div_loss.append(valid_user_div_loss)
        dev_res_div_loss.append(valid_res_div_loss)
        dev_acc_loss.append(valid_acc_loss)
        dev_acc.append(valid_accuracy_metric.result().numpy())   
        print(
            f"Epoch: {epoch + 1}, Valid Loss: {valid_loss}  valid accuracy:{valid_accuracy_metric.result().numpy()}")



        if valid_loss < best_loss_so_far:
            print("find better loss")
            ProtoCNN.save_weights(os.path.join(out_dir, "best_classifier.ckpt"))
            pickle.dump(opt.get_weights(), open(os.path.join(out_dir, "optimizer.pt"), "wb+"))
            best_loss_so_far = valid_loss
        
        
      
       
            test_accuracy_metric.reset_state()
            for x_batch, author_batch, topic_batch, y_batch in tqdm(test_loader):

               
                author_batch = np.asarray(author_batch)
                topic_batch = np.asarray(topic_batch)
                y_batch = np.asarray(y_batch)
    
                #predictions, _, _ = ProtoCNN([x_batch, author_batch, topic_batch], training=False)
                predictions  = ProtoCNN([x_batch, author_batch, topic_batch], training=False)
                test_accuracy_metric.update_state(y_batch, predictions)
    
    
           
            test_acc.append((epoch+1, test_accuracy_metric.result().numpy()))

            print(f"Epoch: {epoch + 1},   test accuracy:{test_accuracy_metric.result().numpy()}")


        with open(out_dir+'train_losses.pkl', 'wb') as file:
            pickle.dump((train_loss, train_acc_loss,train_user_div_loss, train_res_div_loss, train_acc), file)

        with open(out_dir+'valid_losses.pkl', 'wb') as file:
            pickle.dump((dev_loss, dev_acc_loss, dev_user_div_loss, dev_res_div_loss, dev_acc), file)

        with open(out_dir+'test_acc.pkl', 'wb') as file:
            pickle.dump(test_acc, file)

            
            
    
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))  # Optional: set figure size
        plt.plot(train_loss, label='Training Loss')
        plt.plot(dev_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Save the plot as an image
        plt.savefig('training_validation_loss.png')
        
        # Optionally, display the plot
        plt.show()
    
  
