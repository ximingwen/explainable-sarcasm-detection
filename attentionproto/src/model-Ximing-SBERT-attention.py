import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from layer import PrototypeLayer, DistanceLayer, Encoder





class AttentionProtoNet(nn.Module):
    def __init__(self, sequence_length, num_classes, embedding_model, l2_reg_lambda, dropout_keep_prob, k_protos, vect_size):
        super(AttentionProto, self).__init__()
        self.max_l = sequence_length
        self.l2_reg_lambda = l2_reg_lambda
        self.embedding = embedding_model
        #parameters for word-level attention
        self.W_nu = nn.Parameter(torch.randn())
        self.cls_token = nn.Parameter(torch.randn(1, embedding_dim))
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.k_protos = k_protos
        self.vect_size = vect_size
        self.sent_proto_encoder = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100)
      
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.final_dense = nn.Linear(num_classes, activation="softmax")  # Add L2 regularization separately if needed

        self.fc_word_level = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.hidden_shape, bias=True)
        self.W_nu = nn.Parameter(torch.randn(size=(self.hidden_shape, 1)))

        


    def init_prototypelayer(self, sent_cents):
        self.proto_layer = PrototypeLayer(self.k_protos, self.vect_size, sent_cents)

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


    def forward(self, input_content):
        
        x = self.embedding.encode(input_content)  # Adjust based on your embedding_model

        alphas = self.sbert_attention_calculation(x)

        #calculate attention-weighted hidden states
        all_S = []  
        for sample_alphas, sample_hidden in zip(alphas, x):
           S_i = torch.mm(sample_alphas.unsqueeze(0), sample_hidden).squeeze(0)
           all_S.append(S_i)
        x= torch.stack(all_S, dim=0) #batch*num_of_sentence_*embedding_dim
 
        x = x.unsqueeze(0)
        full_distances, protos = self.response_proto_layer(x) #batch*num_of_sentence
        sent_vect = self.res_distance_layer(full_distances) #batch*num_of_sentence

        sent_proto_embedding = sent_proto*protos 

        # add padding

        sent_proto_embedding = torch.cat([self.cls_tokens, sent_proto_embedding], dim=1)  # concatenate cls token with input embeddings
        # Attention

        # z = [batch size, src len, hid dim]

        z = self.sent_proto_encoder(sent_proto_embedding, sent_proto_embedding)
             
        
        combined_vector_final = self.dropout(combined_vector_final)
        scores = self.final_dense(combined_vector_final)

                                  

        return scores

    def embed(self, x):
        # Embedding layer
        x = self.embedding.encode(x)  # Adjust based on how embedding_model is defined
        return x



