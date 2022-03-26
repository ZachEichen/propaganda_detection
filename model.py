import torch 
from transformers import BertForSequenceClassification
import transformers

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
