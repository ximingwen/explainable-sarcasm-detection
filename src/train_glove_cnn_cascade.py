#!/usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
from text_cnn import TextCNN
import os
from tensorflow.contrib import learn
import csv
from time import sleep
import pickle, argparse

from tensorflow.keras import layers, Model, regularizers



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
        dist_hot_vect = e_dist
        return dist_hot_vect


class TextCNN(tf.keras.Model):
    """
    A CNN for text classification in TensorFlow 2.x.
    Uses an embedding layer, followed by a convolutional, max-pooling, and softmax layer.
    """
    def __init__(
        self, sequence_length, num_classes, vocab_size, word2vec_W, word_idx_map, user_embeddings, topic_embeddings,
        embedding_size, batch_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        super(TextCNN, self).__init__()

        self.l2_reg_lambda = l2_reg_lambda
        l2_regularizer = tf.keras.regularizers.l2(l2_reg_lambda)
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, weights=[word2vec_W], input_length=sequence_length, trainable=False)
        self.user_embedding = tf.keras.layers.Embedding(input_dim=user_embeddings.shape[0], output_dim=user_embeddings.shape[1], weights=[user_embeddings], trainable=True)
        self.topic_embedding = tf.keras.layers.Embedding(input_dim=topic_embeddings.shape[0], output_dim=topic_embeddings.shape[1], weights=[topic_embeddings], trainable=True)

        self.conv_layers = []
        for filter_size in filter_sizes:
            conv_layer = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(filter_size, embedding_size), activation='relu')
            self.conv_layers.append(conv_layer)
        

        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.last_dense = tf.keras.layers.Dense(100, activation='relu')
        self.user_topic_dense = tf.keras.layers.Dense(400, activation='relu')
        self.dropout = tf.keras.layers.Dropout(1 - self.dropout_keep_prob)
        # Final dense layer with L2 regularization
        self.final_dense = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2_regularizer)
       

    def call(self, inputs, training=False):
        input_x, input_author, input_topic = inputs
        x = self.embedding(input_x)
        x = tf.expand_dims(x, -1)

        pooled_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x)
            pooled_out = self.max_pool(conv_out)
            pooled_outputs.append(pooled_out)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        h_last = self.last_dense(h_pool_flat)

        user_embeddings = self.user_embedding(input_author)
        topic_embeddings = self.topic_embedding(input_topic)
        combined_vectors = self.concat_layer([h_last, user_embeddings, topic_embeddings])
        combined_vector_final = self.user_topic_dense(combined_vectors)

        
        combined_vector_final = self.dropout(combined_vector_final)

        scores = self.final_dense(combined_vector_final)

        return scores


    def compute_accuracy(self, input_y, scores):
        predictions = tf.argmax(scores, 1, name="predictions")
        correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return accuracy

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
    parser.add_argument("--embedding_dim", type=int, default=300,
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
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch Size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=100,
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
            temp_y = revs[i]['label']
            y.append(temp_y)
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
    
    topic_train = np.asarray(topic_text_id)
    topic_test = np.asarray(test_topic)
    author_train = np.asarray(author_text_id)
    author_test = np.asarray(test_author)
    
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
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
    word_idx_map["@"] = 0
    rev_dict = {v: k for k, v in word_idx_map.items()}
    
    # Training
    # ==================================================
    
 
    cnn = TextCNN(
        sequence_length=max_l,
        num_classes=len(y_train[0]) ,
        vocab_size=len(vocab),
        word2vec_W = W,
        word_idx_map = word_idx_map,
        user_embeddings = user_embeddings,
        topic_embeddings = topic_embeddings,
        embedding_size=args.embedding_dim,
        batch_size=args.batch_size,
        filter_sizes=list(map(int, args.filter_sizes.split(","))),
        num_filters=args.num_filters,
        l2_reg_lambda=args.l2_reg_lambda)

    # Define Training procedure
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=.1e-3)
    criterion = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=keras.losses.Reduction.SUM)

       
    # Define a Checkpoint manager
    checkpoint = tf.train.Checkpoint(model=cnn)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, './checkpoints', max_to_keep=args.num_checkpoints)
    
    # Save the model (checkpoint)
    checkpoint_manager.save()
        
    
    
    def train_step(cnn, x_batch, author_batch, topic_batch, y_batch, optimizer):
        """
        A single training step
        """
        with tf.GradientTape() as tape:
            logits = cnn([x_batch, author_batch, topic_batch], training=True)
            loss_value = cnn.loss(y_batch, logits)
    
        gradients = tape.gradient(loss_value, cnn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))
    
        accuracy_metric.update_state(y_batch, tf.nn.softmax(logits))
        return loss_value, accuracy_metric.result()
    
    def dev_step(cnn, x_batch, author_batch, topic_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        cnn.dropout_keep_prob.assign(1.0)  # Set dropout keep probability to 1.0 for evaluation
    
        logits = cnn([x_batch, author_batch, topic_batch], training=False)
        loss_value = cnn.loss(y_batch, logits)
    
        conf_mat = tf.math.confusion_matrix(
            tf.argmax(y_batch, axis=1), tf.argmax(logits, axis=1), num_classes=num_classes)
    
        return loss_value, conf_mat
            
    
    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, author_train, topic_train, y_train)), args.batch_size, args.num_epochs)
    dev_batches = data_helpers.batch_iter(
        list(zip(x_dev, author_dev, topic_dev, y_dev)), args.batch_size, args.num_epochs)
    # Training loop. For each batch...
    
    train_loss = []
    train_acc = []
    best_acc = 0
    for batch in batches:
        x_batch, author_batch, topic_batch, y_batch = zip(*batch)
        x_batch = np.asarray(x_batch)
        author_batch = np.asarray(author_batch)
        topic_batch = np.asarray(topic_batch)
        y_batch = np.asarray(y_batch)
        t_loss, t_acc = train_step(x_batch, author_batch, topic_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        train_loss.append(t_loss)
        train_acc.append(t_acc)
        if current_step % args.evaluate_every == 0:
            print(current_step)
            print("Train loss {:g}, Train acc {:g}".format(np.mean(np.asarray(train_loss)), np.mean(np.asarray(train_acc))))
            train_loss = []
            train_acc = []
            # Divide into batches
            dev_batches = data_helpers.batch_iter_dev(list(zip(x_dev, author_dev, topic_dev, y_dev)), args.batch_size)
            dev_loss = []
            ll = len(dev_batches)
            conf_mat = np.zeros((2,2))
            for dev_batch in dev_batches:
                x_dev_batch = x_dev[dev_batch[0]:dev_batch[1]]
                author_dev_batch = author_dev[dev_batch[0]:dev_batch[1]]
                topic_dev_batch = topic_dev[dev_batch[0]:dev_batch[1]]
                y_dev_batch = y_dev[dev_batch[0]:dev_batch[1]]
                a, b = dev_step(x_dev_batch, author_dev_batch, topic_dev_batch, y_dev_batch)
                dev_loss.append(a)
                conf_mat += b
            valid_accuracy = float(conf_mat[0][0]+conf_mat[1][1])/len(y_dev)
            print("Valid loss {:g}, Valid acc {:g}".format(np.mean(np.asarray(dev_loss)), valid_accuracy))
            print("Valid - Confusion Matrix: ")
            print(conf_mat)
            test_batches = data_helpers.batch_iter_dev(list(zip(x_test, author_test, topic_test, y_test)), FLAGS.batch_size)
            test_loss = []
            conf_mat = np.zeros((2,2))
            for test_batch in test_batches:
                x_test_batch = x_test[test_batch[0]:test_batch[1]]
                author_test_batch = author_test[test_batch[0]:test_batch[1]]
                topic_test_batch = topic_test[test_batch[0]:test_batch[1]]
                y_test_batch = y_test[test_batch[0]:test_batch[1]]
                a, b = dev_step(x_test_batch, author_test_batch, topic_test_batch, y_test_batch)
                test_loss.append(a)
                conf_mat += b
            print("Test loss {:g}, Test acc {:g}".format(np.mean(np.asarray(test_loss)), float(conf_mat[0][0]+conf_mat[1][1])/len(y_test)))
            print("Test - Confusion Matrix: ")
            print(conf_mat)
            sys.stdout.flush()
            if best_acc < valid_accuracy:
                best_acc = valid_accuracy
                directory = "./models"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                saver.save(sess, directory+'/main_balanced_user_plus_topic', global_step=1)
