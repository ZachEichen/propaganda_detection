import torch 
# from transformers import DistilBertTokenizer, DistilBertModel
import transformers
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 13)

#model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

for param in model.distilbert.parameters():
    param.requires_grad = False
