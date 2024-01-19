import numpy as np
import os
import time
import datetime
import data_helpers
import argparse
import pickle
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow import keras

from matplotlib import pyplot as plt



class TextCNN(Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, pretrained_embeddings, word_idx_map,\
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob):

        
        super(TextCNN, self).__init__()
       
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_size,weights = [pretrained_embeddings],trainable=False)  # Optional: Set to True if you want to fine-tune the embeddings during training

        
        self.convs = []
        for filter_size in filter_sizes:
            conv_block = tf.keras.Sequential([
                layers.Conv2D(num_filters, (filter_size, embedding_size), 
                              padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(sequence_length - filter_size + 1, 1), 
                                    strides=(1,1), padding='valid')])
            self.convs.append(conv_block)

        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout_keep_prob)  # keep_prob will be supplied by call argument
        self.fc = layers.Dense(num_classes, 
                               kernel_regularizer=regularizers.l2(l2_reg_lambda), 
                               activation='softmax')

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
        x = self.dropout(x)
        x = self.fc(x)
        return x, self.fc.weights[0], self.fc.weights[1]

def train_step(x_batch, y_batch):
    """
    A single training step
    """
    with tf.GradientTape() as tape:
        logits, weights, bias = cnn(x_batch)

        weights_l2_loss = tf.nn.l2_loss(weights)
        biases_l2_loss = tf.nn.l2_loss(bias)
        total_l2_loss = weights_l2_loss + biases_l2_loss
     
        prediction_losses = tf.keras.losses.categorical_crossentropy(y_batch, tf.nn.softmax(logits))
        loss = tf.reduce_mean(prediction_losses) + FLAGS.l2_reg_lambda* total_l2_loss

    gradients = tape.gradient(loss, cnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))

    predictions = tf.argmax(logits, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(y_batch, 1))



    return loss, correct_predictions

def dev_step(x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    logits, weights, bias = cnn(x_batch,training = False)

    weights_l2_loss = tf.nn.l2_loss(weights)
    biases_l2_loss = tf.nn.l2_loss(bias)
    total_l2_loss = weights_l2_loss + biases_l2_loss

    prediction_losses = tf.keras.losses.categorical_crossentropy(y_batch, tf.nn.softmax(logits))

    loss = tf.reduce_mean(prediction_losses) +  FLAGS.l2_reg_lambda* total_l2_loss


    predictions = tf.argmax(logits, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(y_batch, 1))
    

    return loss, correct_predictions
    

if __name__ ==  "__main__":
    
    # Parameters
    # ==================================================

    # Data loading params
    parser = argparse.ArgumentParser()

    parser.add_argument("--dev_sample_percentage", default=.1, type=float, help="Percentage of the training data to use for validation")
 

    # Model Hyperparameters
    parser.add_argument("--embedding_dim", default=300, type=int, help="Dimensionality of character embedding (default: 300)")
    parser.add_argument("--filter_sizes", default="3,4,5", type=str, help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument("--num_filters", default=128, type=int, help="Number of filters per filter size (default: 128)")
    parser.add_argument("--dropout_keep_prob", default=0.5, type=float, help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=0.5, type=float, help="L2 regularization lambda (default: 0.5)")

    # Training parameters
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch Size (default: 64)")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs (default: 100)")
    parser.add_argument("--evaluate_every", default=100, type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--checkpoint_every", default=100, type=int, help="Save model after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=5, type=int, help="Number of checkpoints to store (default: 5)")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")

    FLAGS = parser.parse_args()
    
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

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    x_train = np.asarray(x_train)
    x_dev = np.asarray(x_dev)
    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    word_idx_map["@"] = 0
    rev_dict = {v: k for k, v in word_idx_map.items()}
    
   

   

    # Training
    # ==================================================

    cnn = TextCNN(
        sequence_length=max_l,
        num_classes=len(y_train[0]),
        vocab_size=len(W),
        pretrained_embeddings = W,
        word_idx_map = word_idx_map,
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        dropout_keep_prob = FLAGS.dropout_keep_prob)

    # Define Training procedure
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    saver = tf.train.Checkpoint(optimizer=optimizer, model=cnn)
    saver_manager = tf.train.CheckpointManager(saver, checkpoint_dir, max_to_keep=FLAGS.num_checkpoints)


    
    # Generate batches
    # Create training dataset
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   

    # Create testing dataset
    dev_loader = tf.data.Dataset.from_tensor_slices((x_dev, y_dev))
   
    
    # Create testing dataset
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    
    # Training loop. For each batch...
    train_losses = []
    valid_losses = []
    train_accuracys = []
    valid_accuracys = []
    
    best_acc = float('-inf')
    
    
    for epoch in range(FLAGS.num_epochs):
        train_loss = 0
        valid_loss = 0
        correct_predictions_train = None
        correct_predictions_valid = None
       
       
        for x_batch, y_batch in tqdm(train_loader.shuffle(len(train_loader)).batch(FLAGS.batch_size)):
       
            loss, correct_predictions = train_step(x_batch, y_batch)
            train_loss += loss
            if correct_predictions_train is None:
                correct_predictions_train = correct_predictions
            else:
                correct_predictions_train = tf.concat((correct_predictions_train, correct_predictions), axis=0)
        
        train_accuracy = tf.reduce_mean(tf.cast(correct_predictions_train, tf.float32))
        train_accuracys.append(train_accuracy)
        train_losses.append(train_loss)
        
        
        for x_batch, y_batch in tqdm(dev_loader.batch(FLAGS.batch_size)):

            loss, correct_predictions = dev_step(x_batch, y_batch)
            valid_loss += loss
            if correct_predictions_valid is None:
                
                correct_predictions_valid = correct_predictions
            else:
        
                correct_predictions_valid = tf.concat((correct_predictions_valid, correct_predictions), axis=0)
        valid_accuracy = tf.reduce_mean(tf.cast(correct_predictions_valid, tf.float32))
        valid_accuracys.append(valid_accuracy)
        valid_losses.append(valid_loss)
        
        
        if best_acc < valid_accuracy:
            best_acc = valid_accuracy
            save_path = saver_manager.save()
            print("Saved model checkpoint to {}\n".format(save_path))
            
            
        print("epoch: {}, train_loss: {}, train_accuracy: {}, valid_loss:{}, valid_accuracy:{}".format(epoch, train_loss, train_accuracys[-1], valid_loss, valid_accuracys[-1]))
            
            
 
    plt.clf()
    plt.figure()
        
    plt.subplot(211)
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(valid_losses)), valid_losses)
        
    plt.subplot(212)
    plt.plot(range(len(train_accuracys)), train_accuracys)
    plt.plot(range(len(valid_accuracys)), valid_accuracys)
    
    plt.savefig(os.path.join(out_dir, 'loss_graph.png'))
    
    
    correct_predictions_test = None
    
    latest_checkpoint = saver_manager.latest_checkpoint
    saver.restore(latest_checkpoint)
    
    for x_batch, y_batch in tqdm(test_loader.batch(FLAGS.batch_size)):    
        test_loss, correct_predictions = dev_step(x_batch, y_batch)  
        if correct_predictions_test is None:
            correct_predictions_test = correct_predictions
        else:
            correct_predictions_test = tf.concat((correct_predictions_test, correct_predictions), axis=0)
            
    test_accuracy = tf.reduce_mean(tf.cast(correct_predictions_test, tf.float32))
    print("test accuracy {}".format(test_accuracy))