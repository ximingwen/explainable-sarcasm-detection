import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from datetime import datetime
import operator



#customized model
class CustomModel(keras.Model):

    @tf.function
    def train_step(self, data):
        x, y = data
        def pw_distance(A):
            r = tf.reduce_sum(A * A, 1)
            r = tf.reshape(r, [-1, 1])
            D = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
            return D

        def tight_pos_sigmoid_offset(x, offset, e=2.7182818284590452353602874713527):
            return 1 / (1 + tf.math.pow(e, (1 * (offset * x - 0.5))))

        with tf.GradientTape() as tape:

            loss_object = tf.keras.losses.BinaryCrossentropy()
            #the final loss function
            loss = loss_object(y_val, y_pred) + alpha * cost2 + beta * cost3

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}

@property
def metrics(self):
    return [loss_tracker]


class ProtoryNet:

    def __init__():
        self.mappedPrototype = {}

        #create the ProtoryNet:
        # inputs:
        ##+k_cents: the initialized values of prototypes. In the paper, we used KMedoids clustering
        #            to have these values
        ##vect_size: the dimension of the embedded sentence space, if using Google Universal Encoder,
        ##          this value is 512
        ## alpha and beta: the parameters used in the paper, default values are .0001 and .01
    
        loss_tracker = keras.metrics.Mean(name="loss")

        
       

       

    def embed(self,input):
        return self.embModel.predict(input)

    #Evalute the model performance on validation set
    def evaluate(self,x_valid, y):
        right, wrong = 0, 0
        count = 0
        y_preds = []
        for x, y in zip(x_valid, y):
            y_pred = self.model.predict(x)
            y_preds.append(y_pred)
            if count % 500 == 0:
                print('Evaluating y_pred, y ', y_pred, round(y_pred[0]), y)
            if round(y_pred[0]) == y:
                right += 1
            else:
                wrong += 1
            count += 1

        return y_preds, right / (right + wrong)

    #Method to train the model
   
    #this method simple project prototypes to the closest sentences in
    #sample_sent_vects
    def projection(self,sample_sentences,sample_sent_vects,data_size=10000):
        self.prototypes = self.proto_layer.prototypes.numpy()
        d_pos = {}
        #for each prototype
        for p_count, p in enumerate(self.prototypes):
            print('[db] p_count = ', p_count)
            s_count = 0
            d_pos[p_count] = {}
            #find its distances to all sample sentences
            for i, s in enumerate(sample_sent_vects[:data_size]):
                if len(sample_sentences[i]) < 5 or len(sample_sentences[i]) > 100:
                    continue
                d_pos[p_count][i] = np.linalg.norm(sample_sent_vects[i] - p)
                s_count += 1
        #sort those distances, then assign the closest ones to new prototypes
        new_protos = []
        for p_count, p in enumerate(self.prototypes):
            sorted_d = sorted(d_pos[p_count].items(), key=operator.itemgetter(1))
            new_protos.append(sample_sent_vects[sorted_d[0][0]])
        #return these values
        self.prototypes = new_protos
        return new_protos

    #show the list of prototypes
    def showPrototypes(self,sample_sentences,sample_sent_vects,k_protos=10,printOutput=False):
        self.mappedPrototypes = {}
        new_protos = self.projection(sample_sentences,sample_sent_vects)
        data_size = 10000
        d_pos = {}
        for p_count, p in enumerate(new_protos):
            print('p_count = ', p_count)
            s_count = 0
            d_pos[p_count] = {}
            for i, s in enumerate(sample_sent_vects[:data_size]):
                if len(sample_sentences[i]) < 5 or len(sample_sentences[i]) > 100:
                    continue
                d_pos[p_count][i] = np.linalg.norm(sample_sent_vects[i] - p)
                s_count += 1
            print('count = ', s_count)

        k_closest_sents = 10
        recorded_protos_score = {}
        print("Prototypes: ")
        for l in range(k_protos):
            # print("prototype index = ", l)
            recorded_protos_score[l] = {}
            sorted_d = sorted(d_pos[l].items(), key=operator.itemgetter(1))
            for k in range(k_closest_sents):
                i = sorted_d[k][0]
                # print("[db] sorted_d ",sorted_d[0])
                # print("[db] sample_sentences[sorted_d[0][0]]: ",sample_sentences[sorted_d[0][0]])
                self.mappedPrototypes[l] = sample_sentences[sorted_d[0][0]].strip()
                if printOutput:
                    print(sorted_d[k], sample_sentences[i])
            print(self.mappedPrototypes[l])

    #method to manually save the model
    def saveModel(self,name):
        self.model.save_weights(name + ".h5")

    #return the vector value of the input sentence
    def embed(self,input):
        return self.embModel.predict(input)

    #method to generate the number of closest sentences to each prototype
    def protoFreq(self,sample_sent_vect):
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
        print("sorted :",x)

    #re-train the model with new pruned prototype
    def pruningTrain(self,new_k_protos,x_train,y_train,x_test,y_test):
        #print("[db] self prototypes: ",self.prototypes)
        k_cents = self.prototypes[:new_k_protos]
        k_cents = [p.numpy() for p in k_cents]
        #print("[db] k_cents = ",k_cents)
        self.createModel(k_cents=k_cents,k_protos=new_k_protos)
        self.train(x_train,y_train,x_test,y_test)

    # generate the sentence value for each prototype
    # and 10 closest sentences to it
    def showTrajectory(self,input,sample_sentences,sample_vect):
        if len(self.mappedPrototypes) == 0:
            self.showPrototypes(sample_sentences,sample_vect,printOutput=False)
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

        #for small dataset, we use a pretrained sentiment model. We can use any
        #model for sentiment scores
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid_obj = SentimentIntensityAnalyzer()
        print("[db] mappedProtos ", mappedProtos)
        scores = []
        for s in mappedProtos:
            # sentiment_dict = sid_obj.polarity_scores(s)
            scores.append(0.5 + sid_obj.polarity_scores(s)['compound'] / 2)
        return scores

  


def createModel(self,k_cents,k_protos=10,vect_size=512,alpha=0.0001,beta=0.01):
    #building the model
    inputLayer = tf.keras.layers.Input(shape=[], dtype=tf.string)

    l2 = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',
                        trainable=True)(inputLayer)
    seqEncoder = tf.expand_dims(l2, axis=0)

   
  
    full_distances, protos = self.proto_layer(seqEncoder)
    dist_hot_vect = self.distance_layer(full_distances)

    RNN_CELL_SIZE = 128
    lstmop, forward_h, forward_c = LSTM(RNN_CELL_SIZE, return_sequences=True, return_state=True)(dist_hot_vect)

    z1 = tf.keras.layers.Dense(1, activation='sigmoid')(lstmop[:, -1, :])
    z = tf.squeeze(z1, axis=0)

    model = CustomModel(inputLayer, z)

    for l in model.layers:
        if "proto_layer" in l.name:
            protoLayerName = l.name
        if "distance_layer" in l.name:
            distanceLayerName = l.name

    #protoLayer = model.get_layer(protoLayerName)
    #distLayer = model.get_layer(distanceLayerName)

    print("[db] model.input = ", model.input)
    print("[db] protoLayerName = ", protoLayerName)
    print("[db] protoLayer = ", protoLayer)
    print("[db] protoLayer.output = ", protoLayer.output)
    print("[db] distanceLayer.output = ", distLayer.output)
   # auxModel = keras.Model(inputs=model.input,
                           outputs=protoLayer.output)

    #auxModel1 = keras.Model(inputs=model.input,
                            outputs=distLayer.output)

    #auxModel2 = keras.Model(inputLayer, z)

    # auxOutput = auxModel(l1)
    #model.auxModel = auxModel
    #model.auxModel1 = auxModel1
    #self.auxModel2 = auxModel2
    #self.embModel = keras.Model(inputLayer,l2)
    model.summary()

    self.model = model
    return model



#predict the sentiment score of any input string
def predict(self,input):
    return self.model.predict(input)
