import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import json

nltk.download('punkt')


class SarcasmDetection(Dataset):
    def __init__(self, texts, labels,  model_name, max_sentences=4, max_tokens=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.texts = texts
        self.labels = labels
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sentences = sent_tokenize(text)  # [list]:sentences
       
        if len(sentences) > self.max_sentences:
            sentences = text[-self.max_sentences:]
    
        tokenized_sentences = []
        attention_masks = []

        for sentence in sentences:
            # Tokenize sentences
            encoded_dict = self.tokenizer(sentence, truncation=True, return_tensors='pt', padding='max_length', return_attention_mask=True, max_length= self.max_tokens)

            #add_special_tokens=True,
   
            tokenized_sentences.append(encoded_dict['input_ids'].squeeze(0))
            attention_masks.append(encoded_dict['attention_mask'].squeeze(0))

   
        while len(tokenized_sentences) < self.max_sentences:
            tokenized_sentences.append(torch.zeros(self.max_tokens, dtype=torch.long))
            attention_masks.append(torch.zeros(self.max_tokens, dtype=torch.long))

        

        tokenized_sentences = torch.stack(tokenized_sentences)
        attention_masks = torch.stack(attention_masks)
        label = torch.tensor(label)

        return tokenized_sentences, attention_masks, label


def custom_collate_fn(batch):
    # 'batch' is a list of tuples where each tuple is the output of __getitem__
    # Each tuple contains tokenized_sentences and attention_masks for a single item

    # Unzip the batch into separate lists for tokenized sentences and attention masks
    tokenized_sentences_batch, attention_masks_batch, labels = zip(*batch)

    # Stack the tokenized sentences and attention masks to create batched tensors
    tokenized_sentences_batch = torch.stack(tokenized_sentences_batch)
    attention_masks_batch = torch.stack(attention_masks_batch)
    label_batch = torch.stack(labels)



    return {'input_ids': tokenized_sentences_batch, 'attention_mask': attention_masks_batch}, label_batch










# class DataLoader:
#     def __init__(self, data, batch_size=200, shuffle=True):
#         self.data = data
#         self.batch_size = batch_size
#         self.shuffle = shuffle

        
    

#     def __len__(self):
#         # Returns the number of batches
#         return int(np.ceil(len(self.data) / self.batch_size))

#     def __iter__(self):
#         # Shuffles the indexes if required
#         data = pd.DataFrame(self.data).to_numpy()
#         data_size = len(data)
#         num_batches_per_epoch = int((len(data)-1)/self.batch_size) + 1
      
#         if self.shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
        
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * self.batch_size
#             end_index = min((batch_num + 1) * self.batch_size, data_size)
#             output = list(zip(*shuffled_data[start_index:end_index]))
#             yield output[0],  output[1]