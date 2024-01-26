
class AttentionProtoNet(nn.Module):
    def __init__(self, sequence_length, num_classes, embedding_model, user_embeddings, topic_embeddings, embedding_size, filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):
        super(AttentionProto, self).__init__()
        self.max_l = sequence_length
        self.l2_reg_lambda = l2_reg_lambda
        self.embedding = embedding_model
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.k_protos = k_protos
        self.vect_size = vect_size
        self.sent_proto_encoder = (input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100)
      
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.final_dense = nn.Linear(num_classes, activation="softmax")  # Add L2 regularization separately if needed
        


    def init_prototypelayer(self, res_cents, user_cents):
        self.response_proto_layer = PrototypeLayer(self.k_protos, self.vect_size, res_cents)

    def sbert_attention_calculation(self, hidden_states):
       #hidden states expected to be Batch X Length X Hidden
       upsilon = torch.tanh(self.fc_word_level(hidden_states)) #still Batch X Length X Hidden
    
       nu_stack = []
       for i in range(upsilon.shape[0]): #for each instance in batch
           nu_i = torch.mv(upsilon[i], self.W_nu.squeeze()) #calculate nu_t for each step, i call all the nu_t's together nu_i, as in nu's for the instance
           nu_stack.append(nu_i)
       nu_stack = torch.stack(nu_stack, dim=0) #Batch X Length
       alphas = F.softmax(nu_stack, dim=1) #Batch X Length
    
     
       return alphas


    def forward(self, input_content, input_author, input_topic):
        
        x = self.embedding.encode(input_content)  # Adjust based on your embedding_model



          #calculate attention-weighted hidden states
       all_S = []
       for sample_alphas, sample_hidden in zip(alphas, hidden_states):
           S_i = torch.mm(sample_alphas.unsqueeze(0), sample_hidden).squeeze(0)
           all_S.append(S_i)
       all_S = torch.stack(all_S, dim=0)
    
       all_sims = []
       for S_i in all_S:
           diff_i = S_i - self.prototypes
           diff_i_sqrd = diff_i.pow(2)
           diff_i_summed = diff_i_sqrd.sum(dim=1)
           sim_i = 1 / (diff_i_summed.sqrt() + self.hparams.dist_eps)
           all_sims.append(sim_i)
       all_sims = torch.stack(all_sims, dim=0) #Batch x num_prototypes

        

        x = x.unsqueeze(0)
        full_distances, res_protos = self.response_proto_layer(x)
        sent_vect = self.res_distance_layer(full_distances)

        sent_proto_embedding = sent_proto*proto

        sent_proto_embedding = torch.cat([self.cls_tokens, sent_proto_embedding], dim=1)  # concatenate cls token with input embeddings
        # Attention

        # z = [batch size, src len, hid dim]

        z = self.sent_proto_encoder(sent_proto_embedding, sent_proto_embedding)
             
        
        combined_vector_final = self.dropout(combined_vector_final)
        scores = self.final_dense(combined_vector_final)

                                  

        return 

    def embed(self, x):
        # Embedding layer
        x = self.embedding.encode(x)  # Adjust based on how embedding_model is defined
        return x
