
from torch import nn
import torch
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel,BertSelfAttention
import logging
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.nn.functional as F

logger=logging.getLogger(__name__)

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:,1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertTorchClassfication(nn.Module):
    '''
    Bert CLS Model
    '''
    base_model_prefix = "bert"

    def __init__(self,bert_model_path, config):
        super(BertTorchClassfication, self).__init__()
        self.bert=BertModel.from_pretrained(bert_model_path, config=config)
        #self.pooler=BertPooler(config)
        #self.bert_attention=BertSelfAttention(config)

    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None,position_ids=None,
                index1=None,index2=None):
        '''
        :param input_ids: 使用tokenizer将输入文本转化为token_ids
        :param attention_mask: 0和1组成的mask向量，让模型知道应该学习哪一部分
        :param token_type_ids: segment向量
        :param labels: 标签
        :param position_ids:位置信息向量
        :param index1: 需要做attention的文本1位置信息
        :param index2: 需要做attention的文本2位置信息
        :return:
        '''
        outputs=self.bert.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids)
        cls_representation = outputs.last_hidden_state[:, 0, :]

        '''
        #如果需要做attention的话，使用该部分,并且将线性回归的参数config.hidden_size修改为相应的大小*n
        all_data=outputs[0]#句子全部输出/
        attention_1=torch.index_select(all_data,dim=1,index=index1)
        attention_2=torch.index_select(all_data,dim=1,index=index2)
        extra_attention_output=self.bert_attention.forward(hidden_states=attention_1,
                                                           encoder_hidden_states=attention_2,
                                                           attention_mask=attention_mask)[0]
        extra_attention=torch.mean(extra_attention_output.float(),dim=1)#将得到的attention做平均
        output=torch.cat((pooled_output,extra_attention),dim=1)#将tensor拼接
        '''
        return cls_representation
    
    
class PrototypeLayer(nn.Module):
    def __init__(self, input_dim, prototype_dim, num_prototypes,threshold, loss_weights):
        super(PrototypeLayer, self).__init__()
        self.num_prototypes = num_prototypes
        self.num_positive_prototypes = num_prototypes // 2  # 正类原型的数量
        self.num_negative_prototypes = num_prototypes - self.num_positive_prototypes  # 负类原型的数量
        self.positive_prototype_indices = torch.arange(self.num_positive_prototypes)  # 初始化正类和负类原型索引
        self.negative_prototype_indices = torch.arange(self.num_positive_prototypes, self.num_prototypes)
        self.prototype_vectors = nn.Parameter(torch.randn(num_prototypes, prototype_dim))  # 初始化原型向量
        self.linear = nn.Linear(input_dim,prototype_dim)
        self.threshold = threshold
        self.loss_weights = loss_weights 

    def forward(self, x, labels):
        z = self.linear(x)
        
        # 计算余弦相似度
        cosine_similarities = F.cosine_similarity(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), dim=-1)

        # 将相似度乘以原型向量，得到每个 token 的表示
        token_sequences = cosine_similarities.unsqueeze(2) * self.prototype_vectors.unsqueeze(0)
        
        loss = self.loss_weights['diversity_loss_z_p'] * self.diversity_loss_z_p(z) + self.loss_weights['diversity_loss_p_z'] * self.diversity_loss_p_z(z) + self.loss_weights['diversity_loss_p_p'] * self.diversity_loss_p_p() + self.loss_weights['loss_num_prototypes'] * self.loss_num_prototypes(z) + self.loss_weights['cluster_loss'] * self.cluster_loss(z, labels, cosine_similarities) + self.loss_weights['seperation_loss'] * self.cluster_loss(z, labels, cosine_similarities)
        
        return token_sequences, loss
    
    def diversity_loss_z_p(self,z):
        # 假设 prototype_vectors 是原型向量，z 是样本投影
        # prototype_vectors 的形状是 (num_prototypes, prototype_dim)
        # z 的形状是 (batch_size, prototype_dim)
        # 对于每个prototype，寻找最近的z，距离取平均

        # 计算每个样本与每个原型之间的距离
        distances = torch.cdist(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), p=2)  # 形状变为 (batch_size, num_prototypes)

        # 找到每个原型最近的样本的索引
        nearest_indices = torch.argmin(distances, dim=0)

        # 根据索引找到每个原型最近的样本
        nearest_samples = z[nearest_indices]

        # 计算每个样本与其对应原型之间的距离
        distances_per_prototype = torch.norm(nearest_samples - self.prototype_vectors, dim=-1)

        # 对所有原型的距离取平均
        loss = distances_per_prototype.mean()
        
        return loss
    
    def diversity_loss_p_z(self,z):
        # 假设 z 是模型输出的表示，prototype_vectors 是原型向量
        # z 的形状是 (batch_size, prototype_dim)
        # prototype_vectors 的形状是 (num_prototypes, prototype_dim)
        # 对于每个z，寻找最近的prototype，距离取平均

        # 计算每个样本 z 与每个原型之间的距离
        distances = torch.cdist(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), p=2)  # 形状变为 (batch_size, num_prototypes)

        # 找到每个样本 z 最近的原型的索引
        nearest_indices = torch.argmin(distances, dim=1)

        # 根据索引找到每个样本 z 最近的原型
        nearest_prototypes = self.prototype_vectors[nearest_indices]

        # 计算每个样本与其最近原型之间的距离
        distances_per_sample = torch.norm(z - nearest_prototypes, dim=-1)

        # 对所有样本的距离取平均
        loss = distances_per_sample.mean()
        
        return loss
    
    def diversity_loss_p_p(self):
        # 假设 prototype_vectors 是原型向量
        # prototype_vectors 的形状是 (num_prototypes, prototype_dim)
        # 计算prototype间距离的平均值

        # 计算每两个原型之间的距离
        distances = torch.cdist(self.prototype_vectors, self.prototype_vectors, p=2)  # 形状变为 (num_prototypes, num_prototypes)

        # 将对角线元素设为一个很大的值，以避免将原型与自身的距离纳入平均计算中
        torch.fill_diagonal_(distances, float('inf'))

        # 计算原型间距离的平均值
        average_distance = distances.mean()
        
        return average_distance
    
    def loss_num_prototypes(self,z):
        # 假设 z 是模型输出的表示，prototype_vectors 是原型向量
        # z 的形状是 (batch_size, prototype_dim)
        # prototype_vectors 的形状是 (num_prototypes, prototype_dim)

        # 计算每个样本 z 与每个原型之间的相似度
        similarities = F.cosine_similarity(z.unsqueeze(1), self.prototype_vectors.unsqueeze(0), dim=-1)  # 形状为 (batch_size, num_prototypes)

        # 计算相似度大于阈值 threshold 的原型数量
        num_similar_prototypes = (similarities > self.threshold).sum(dim=1)

        # 计算超过总原型数量的 1/4 的惩罚原型数量
        penalize_prototypes = (num_similar_prototypes > self.num_prototypes / 4).float()

        # 计算惩罚项的损失
        penalty_loss = penalize_prototypes.sum()
        
        return penalty_loss
    
    def cluster_loss(self, z, labels, cosine_similarities):
        # 假设 z 是模型输出的表示，prototype_vectors 是原型向量
        # z 的形状是 (batch_size, representation_dim)
        # prototype_vectors 的形状是 (num_prototypes, representation_dim)
        # positive_indices 是属于正类的样本的索引
        # negative_indices 是属于负类的样本的索引
        # positive_prototype_indices 是属于正类的prototype索引
        # negative_prototype_indices 是属于负类的prototype索引
        
        # 找到所有正类样本的索引
        positive_indices = torch.nonzero(labels == 1, as_tuple=True)[0]
        
        # 提取所有正类样本与所有正类原型之间的相似度
        positive_prototype_similarities = cosine_similarities[positive_indices[:, None], self.positive_prototype_indices]
        
        # 计算每个样本与其最近的属于正类的原型之间的距离
        min_positive_prototype_similarities, _ = positive_prototype_similarities.max(dim=1)  # 每个样本的最大相似度
        positive_distances = 1 - min_positive_prototype_similarities  # 计算每个样本与其最近的属于正类的原型之间的距离
        
        # 找到所有负类样本的索引
        negative_indices = torch.nonzero(labels == 0, as_tuple=True)[0]  # 负类样本的索引
        
        # 提取所有负类样本与所有负类原型之间的相似度
        negative_prototype_similarities = cosine_similarities[negative_indices[:, None], self.negative_prototype_indices]
        
        # 计算每个样本与其最近的属于负类的原型之间的距离
        min_negative_prototype_similarities, _ = negative_prototype_similarities.max(dim=1)  # 每个样本的最大相似度
        negative_distances = 1 - min_negative_prototype_similarities  # 计算每个样本与其最近的属于负类的原型之间的距离
        
        # 计算所有样本的距离取平均，作为损失
        total_distances = torch.cat([positive_distances, negative_distances], dim=0)  # 将正类和负类样本的距离拼接起来，形状为 (batch_size * 2, num_prototypes)
        average_loss = total_distances.mean()  # 对所有样本的距离取平均，作为损失
        
        return average_loss
    
    def seperation_loss(self, z, labels, cosine_similarities):
        # 假设 z 是模型输出的表示，prototype_vectors 是原型向量
        # z 的形状是 (batch_size, representation_dim)
        # prototype_vectors 的形状是 (num_prototypes, representation_dim)
        # positive_indices 是属于正类的样本的索引
        # negative_indices 是属于负类的样本的索引
        # positive_prototype_indices 是属于正类的prototype索引
        # negative_prototype_indices 是属于负类的prototype索引
        
        
        # 找到所有正类样本的索引
        positive_indices = torch.nonzero(labels == 1, as_tuple=True)[0]
        
        # 提取所有正类样本与所有负类原型之间的相似度
        positive_prototype_similarities = cosine_similarities[positive_indices[:, None], self.negative_prototype_indices]
        
        # 计算每个样本与其最近的属于正类的原型之间的距离
        min_positive_prototype_similarities, _ = positive_prototype_similarities.max(dim=1)  # 每个样本的最大相似度
        positive_distances = min_positive_prototype_similarities  # 计算每个样本与其最近的属于负类的原型之间的距离
        
        # 找到所有负类样本的索引
        negative_indices = torch.nonzero(labels == 0, as_tuple=True)[0]  # 负类样本的索引
        
        # 提取所有负类样本与所有负类原型之间的相似度
        negative_prototype_similarities = cosine_similarities[negative_indices[:, None], self.positive_prototype_indices]
        
        # 计算每个样本与其最近的属于负类的原型之间的距离
        min_negative_prototype_similarities, _ = negative_prototype_similarities.max(dim=1)  # 每个样本的最大相似度
        negative_distances = min_negative_prototype_similarities  # 计算每个样本与其最近的属于负类的原型之间的距离
        
        # 计算所有样本的距离取平均，作为损失
        total_distances = torch.cat([positive_distances, negative_distances], dim=0)  # 将正类和负类样本的距离拼接起来，形状为 (batch_size * 2, num_prototypes)
        average_loss = total_distances.mean()  # 对所有样本的距离取平均，作为损失
        
        return average_loss        
        


    
class TransformerLayer(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, hidden_dim, dropout):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout)
        self.num_layers = num_layers
        self.transformer_encoder = nn.TransformerEncoder(self.transformer, num_layers)

    def forward(self, x):
        # Permute dimensions from (batch_size, sequence_length, embedding_dim) to (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        
        # Add [CLS] token to the beginning of each sequence
        cls_token = torch.zeros(1, x.size(1), x.size(2)).to(x.device)
        x_with_cls = torch.cat((cls_token, x), dim=0)
        
        # Apply transformer encoder
        output_with_cls = self.transformer_encoder(x_with_cls)
        
        # Extract [CLS] token representation
        cls_representation = output_with_cls[0, :, :]  # [CLS] token is at position 0
        
        return cls_representation


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))  # 输入层
        self.layers.append(nn.ReLU())  # 第一个隐藏层后面的ReLU激活函数
        for _ in range(num_layers - 1):  # 添加额外的隐藏层和激活函数
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, output_dim))  # 输出层
        self.layers.append(nn.Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Bert_Prototype(nn.Module):
    def __init__(self, bert_model_path, config, bert_cls_dim, prototype_dim, num_prototypes, prototype_threshold, prototype_loss_weights, transformer_dim, transformer_layers, transformer_heads, transformer_dropout, fc_output_dim, mlp_hidden_dim, mlp_output_dim, mlp_num_layers, mlp_dropout):
        super(Bert_Prototype, self).__init__()
        self.bert_cls = BertTorchClassfication(bert_model_path, config)
        for param in self.bert_cls.parameters():
            param.requires_grad = False  # 将BERT模型的参数设置为不可训练
        self.prototype_layer = PrototypeLayer(bert_cls_dim, prototype_dim,num_prototypes, prototype_threshold, prototype_loss_weights)
        self.transformer_layer = TransformerLayer(prototype_dim, transformer_layers, transformer_heads, transformer_dim, transformer_dropout)
        self.fc_layer = FullyConnectedLayer(transformer_dim, fc_output_dim)
        self.mlp = MLP(fc_output_dim, mlp_hidden_dim, mlp_output_dim, mlp_num_layers, mlp_dropout)

    def forward(self, x):
        bert_output = self.bert_cls(x)
        prototype_output, prototype_loss = self.prototype_layer(bert_output)
        transformer_cls_token =  self.transformer_layer(prototype_output)# 提取Transformer输出的CLS token
        fc_output = self.fc_layer(transformer_cls_token)
        mlp_output = self.mlp(fc_output)
        return mlp_output, prototype_loss



        
