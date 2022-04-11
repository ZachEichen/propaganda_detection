import csv
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


class ProppyDataset(Dataset):
    """Dataset for Proppy dataset (binary classification)"""

    def __init__(self, filepath: Path):
        super().__init__()
        if not filepath.is_file():
            raise Exception("Invalid filepath to tsv file")
        self.data = []
        with open(filepath, newline='', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, dialect=csv.excel_tab)
            for row in tqdm(reader):
                article_text = row[0]
                propaganda_label = int(row[14])
                self.data.append({
                    "text": article_text,
                    "label": float(propaganda_label == 1),
                })
        self.encoding = tokenizer([x['text'] for x in self.data], return_tensors='pt', padding=True, truncation=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['label'] = self.data[idx]['label']
        # item = {"encoding":self.encoding[idx],
        #         "labels"  :self.data[idx]['labels']}
        return item


class FallacyDataset(Dataset):
    """Dataset for Proppy dataset (binary classification)"""

    def __init__(self, filepath: Path):
        super().__init__()
        if not filepath.is_file():
            raise Exception("Invalid filepath to csv file")
        data_df = pd.read_csv(filepath)
        if not ("updated_label" in data_df.columns and "source_article" in data_df.columns):
            raise Exception("could not find correct columns in input")
        # create fallacy <--> labelnum mapping
        self.fallacy_list = list(data_df['updated_label'].unique())
        self.fallacy_dict = {fallacy:i for i, fallacy in enumerate(self.label_to_fallacy)}
        data_df['label'] = data_df["updated_label"].apply(lambda x: self.fallacy_dict[x])
        data_df = data_df[~data_df.label.astype(int).gt(12)]

        self.data = (data_df[["source_article","updated_label",'label']]
                .rename(columns= {"source_article":"text","updated_label":"fallacy"})
                .to_dict(orient="records")
            )
        self.encoding = tokenizer([x['text'] for x in self.data], return_tensors='pt', padding=True, truncation=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['text'] = self.data[idx]['text']
        item['label'] = self.fallacy_dict[self.data[idx]['fallacy']]
        return item    

    @property
    def label_to_fallacy(self):
        return self.fallacy_list

    @property
    def fallacy_to_label(self):
        return self.fallacy_dict
