import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from layer import PrototypeLayer, DistanceLayer, Encoder
from transformers import AutoModel

class AttentionProtoNet(nn.Module):
    def __init__(self, sequence_length, num_classes, embedding_model, l2_reg_lambda, dropout_keep_prob, k_protos, embedding_dim, num_of_sentence, n_layers, n_heads, pf_dim, encoder_dropout, device):
        super(AttentionProtoNet, self).__init__()
        self.device = device
        self.max_l = sequence_length
        self.l2_reg_lambda = l2_reg_lambda
       
        self.embedding_model = AutoModel.from_pretrained(embedding_model)

        # #parameters for word-level attention
        # self.fc_word_level = torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=True)
        # self.W_nu = nn.Parameter(torch.randn(size=(embedding_dim, 1)))
        
     
        self.k_protos = k_protos
        self.vect_size = embedding_dim
        self.sent_proto_encoder = Encoder(self.vect_size, n_layers, n_heads, pf_dim, encoder_dropout, device, max_length=4*k_protos+1)
      
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        self.final_dense = nn.Linear(768,num_classes)  # Add L2 regularization separately if needed

        self.sent_distance_layer = DistanceLayer()
        


    def init_prototypelayer(self, res_cents):
        self.sent_proto_layer = PrototypeLayer(self.k_protos, self.vect_size, res_cents, self.device)

    def sbert_attention_calculation(self, hidden_states):
       #hidden states expected to be Batch X Length X Hidden
       upsilon = torch.tanh(self.fc_word_level(torch.tensor(hidden_states))) #still Batch X Length X Hidden
    
       nu_stack = []
       for i in range(upsilon.shape[0]): #for each instance in batch
           nu_i = torch.mv(upsilon[i], self.W_nu.squeeze()) #calculate nu_t for each step, i call all the nu_t's together nu_i, as in nu's for the instance
           nu_stack.append(nu_i)
       nu_stack = torch.stack(nu_stack, dim=0) #Batch X Length
       alphas = F.softmax(nu_stack, dim=1) #Batch X Length
    
     
       return alphas


    def forward(self, inputs):

        #input_text: batch_size*num_of_sentence*num_of_token
        
        #Compute token embeddings
       
        batch_size, num_of_sent, num_of_token = inputs["input_ids"].shape
        

        outputs = self.embedding_model(input_ids=inputs["input_ids"].view(-1, 100).to(self.device) , attention_mask=inputs["attention_mask"].view(-1, 100).to(self.device) )  

        # Get the last hidden states
        last_hidden_states = outputs.last_hidden_state
        
        # Extract the [CLS] token's embeddings
        cls_embeddings = last_hidden_states[:, 0, :]
        x = cls_embeddings.view(batch_size, num_of_sent,-1)
        
        # #Perform pooling. In this case, mean pooling
        # sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # encoded_input = model.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # with torch.no_grad():
        #     model_output = model[0](**encoded_input)

        # embeddings = model_output.last_hidden_state
        
        # # Decode tokens to words (for demonstration)
        # decoded_tokens = [self.embedding.tokenizer.decode([t]) for t in token_ids]
        
        # x = self.embedding.encode(input_content)  # Adjust based on your embedding_model

        # print("shape ", x.shape) #2 *384

        # alphas = self.sbert_attention_calculation(x)

        # #calculate attention-weighted hidden states
        # all_S = []  
        # for sample_alphas, sample_hidden in zip(alphas, x):
        #    S_i = torch.mm(sample_alphas.unsqueeze(0), sample_hidden).squeeze(0)
        #    all_S.append(S_i)
        # x= torch.stack(all_S, dim=0) #batch*num_of_sentence_*embedding_dim

     
 
       
        #  ([ 60, 4, 768])
        full_distances, prototypes = self.sent_proto_layer(x) #batch*num_of_sentence
        sent_vect = self.sent_distance_layer(full_distances) #batch*num_of_sentence

        # Reshape x for broadcasting
        x_expanded = sent_vect.unsqueeze(-1)  # Shape becomes [batch_size, num_sentence, num_proto, 1]
        
        # Expand prototypes for broadcasting
        prototypes_expanded = prototypes.unsqueeze(0).unsqueeze(0)  # Add batch and sentence dimensions  1, 1, num_of_proto, proto_dim
        prototypes_expanded = prototypes_expanded.expand(batch_size, num_of_sent, self.k_protos, self.vect_size)   #batch_size, num_sent, num_proto,proto_dim
        
        # Element-wise multiplication
        result = x_expanded * prototypes_expanded  # Shape: [batch_size, num_sentence, num_proto, proto_dim]
        sent_proto_embedding= result.view(batch_size, -1, self.vect_size)

       

        # add padding
        cls_tokens = torch.zeros(1,  self.vect_size).to(self.device)

        # Reshape it to [1, 1, 768] to align for concatenation
        cls_tokens = cls_tokens.unsqueeze(1)
        
        # Repeat the tensor_to_add 60 times along the 0th dimension to match original_tensor's batch size
        cls_tokens = cls_tokens.repeat(batch_size, 1, 1)
        
        # Concatenate the tensors along the second dimension
        sent_proto_embedding = torch.cat((cls_tokens, sent_proto_embedding), dim=1)
        
        # Attention
        # Example attention mask tensor (replace with your actual tensor)

        # z = [batch size, src len, hid dim]
        sentence_mask = inputs["attention_mask"].sum(dim=2) != 0  # Shape: [batch_size, num_of_sentence]

        # Step 2: Broadcast the sentence mask
        # Repeat for each token and each prototype
        sentence_mask = sentence_mask.unsqueeze(-1).repeat(1, 1, self.k_protos) #  Shape: [batch_size, num_of_sentence,  k]
        
        # Step 3: Concatenate the sentence masks
        # Reshape to concatenate along the sentence*token dimension
        sentence_mask = sentence_mask.view(batch_size, num_of_sent * self.k_protos) # Shape: [Shape: [batch_size, num_of_sentence * k]
        cls_tokens_mask = torch.ones(batch_size ,1) #batch_size*1*dim
        concatenated_mask =  torch.cat((cls_tokens_mask, sentence_mask ), dim=1).unsqueeze(1).unsqueeze(1).to(self.device)  

    
        z = self.sent_proto_encoder(sent_proto_embedding, concatenated_mask) #batch_size,65,dim    batch_size,1,1,65

        #output [batch_size,src_length,dim] 
             
        
        combined_vector_final = self.dropout(z[:,0,:])
        scores = self.final_dense(combined_vector_final)

                                  

        return scores

    def embed(self, inputs):
        # Embedding layer
        # print(inputs["input_ids"].shape)
        # print(inputs["attention_mask"].shape)

        batch_size, num_of_sent, num_of_token = inputs["input_ids"].shape
        

        outputs = self.embedding_model(input_ids=inputs["input_ids"].view(-1, 100).to(self.device) , attention_mask=inputs["attention_mask"].view(-1, 100).to(self.device) )  

        # Get the last hidden states
        last_hidden_states = outputs.last_hidden_state
        
        # Extract the [CLS] token's embeddings
        cls_embeddings = last_hidden_states[:, 0, :]
       

       
        return cls_embeddings

