import torch
from transformers import DistilBertForSequenceClassification

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=13)

for param in model.distilbert.parameters():
    param.requires_grad = False
