class TextCNN(tf.keras.Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def init(self, sequence_length, num_classes, vocab_size, word2vec_W, word_idx_map, user_embeddings, topic_embeddings,embedding_size, batch_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        super(TextCNN, self).__init__()

        # Embedding layer
        self.W = tf.Variable(word2vec_W, name="W")
        self.user_W = tf.Variable(user_embeddings, name='user_W')
        self.topic_W = tf.Variable(topic_embeddings, name='topic_W')
        self.embedded_chars = tf.keras.layers.Embedding(vocab_size, embedding_size, weights=[self.W])(inputs)
        self.user_embedding_vectors = tf.keras.layers.Embedding(vocab_size, embedding_size, weights=[self.user_W])(inputs)
        self.topic_embedding_vectors = tf.keras.layers.Embedding(vocab_size, embedding_size, weights=[self.topic_W])(inputs)

        # Create a convolution + maxpool layer for each filter size
        self.convolutions = []
        for filter_size in filter_sizes:
            conv = tf.keras.layers.Conv2D(num_filters, (filter_size, embedding_size), activation='relu')(self.embedded_chars)
            pool = tf.keras.layers.GlobalMaxPooling2D()(conv)
            self.convolutions.append(pool)

        # Combine all the pooled features
        self.h_pool = tf.concat(self.convolutions, axis=1)
        self.h_pool_flat = tf.keras.layers.Flatten()(self.h_pool)

        # Add another layer
        self.h_last = tf.keras.layers.Dense(units=100, activation='relu')(self.h_pool_flat)

        # Add user embeddings
        self.combined_vectors = tf.concat([self.h_last, self.user_embedding_vectors], axis=1)
        self.combined_vectors = tf.concat([self.combined_vectors, self.topic_embedding_vectors], axis=1)
        self.final_vector = tf.keras.layers.Dense(units=400, activation='relu')(self.combined_vectors)

        # Final (unnormalized) scores and predictions
        self.scores = tf.keras.layers.Dense(units=num_classes)(self.final_vector)
        self.predictions = tf.argmax(self.scores, axis=1)

    def call(self, inputs, training=False):
        return self.predictions, self.scores
    
