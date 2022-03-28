import torch 
from transformers import BertForSequenceClassification
import transformers

# changed from bert-base-uncased to match base paper more closely 
model = BertForSequenceClassification.from_pretrained("microsoft/deberta-base")
