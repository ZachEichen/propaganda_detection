import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path
import csv

from transformers import BertTokenizerFast

class ProppyDataset(Dataset):
    """Dataset for Proppy dataset (binary classification)"""

    def __init__(self, filepath: Path, device="cuda"):
        super().__init__()
        self.data = []
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        with open(filepath, newline='') as tsvfile:
            reader = csv.reader(tsvfile, dialect=csv.excel_tab)
        for row in tqdm(reader):
            article_text = row[0]
            propaganda_label = int(row[14])
            self.data.append({
                   "text": article_text,
                   "label": propaganda_label,
            })
    
        self.encoding = tokenizer([x['text'] for x in self.data], return_tensors='pt', padding=True, truncation=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.encoding[idx]
