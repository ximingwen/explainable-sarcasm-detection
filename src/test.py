from transformers import BertTokenizer, TFBertModel

bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = TFBertModel.from_pretrained(bert_model_name)

data = ["I am happy oooo","xxxxxxxxx"]
test_x = tokenizer(data, return_tensors="tf", padding=True, truncation=True, max_length=10 )
print(bert_model(input_ids = test_x["input_ids"], attention_mask= test_x["attention_mask"])[0][:,0,:].shape)