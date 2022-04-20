import csv
from pathlib import Path
from random import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ProppyDatasetEmb(Dataset):
    """Bert-preprocessed dataset for proppy corpus """

    def __init__(self, filepath: Path,embspath:Path,balance=False):
        super().__init__()
        
        if  not filepath.is_file():
            raise Exception("Invalid filepath to tsv file")
        #initialize distilbert from pretrained 
        self.data = []
        wherelist = []
        with open(filepath, newline='') as tsvfile:
            reader = csv.reader(tsvfile, dialect=csv.excel_tab)
            for row in tqdm(reader):
                article_text = row[0]
                prop_label = float(row[14])
                # print(embedding[0].shape)
                # print(embedding.keys())
                wherelist.append((not balance)
                                  or prop_label > 0 
                                  or random() < .125)
                if wherelist[-1]: 
                    self.data.append({
                        "label": int(prop_label ==1) ,
                    })   
        wherelist = torch.Tensor(wherelist).bool()
        self.embeds = torch.load(embspath).float()[wherelist,:]
        
        if not (self.embeds.shape[0] == len(self.data)): 
            print(self.embeds.shape) 
            print(len(self.data))
            assert(self.embeds.shape[0] == len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {"label": self.data[idx]["label"], 
                "feats": self.embeds[idx,:]}
        return item    


class FallacyDatasetEmb(Dataset):
    """Bert-preprocessed dataset for Logical fallacy dataset """

    def __init__(self, filepath: Path,embeds_path:Path,label_header = "updated label"):
        super().__init__()
        if not filepath.is_file():
            raise Exception("Invalid filepath to csv file")
        data_df = pd.read_csv(filepath)
        if not (label_header in data_df.columns and "source_article" in data_df.columns):
            raise Exception("could not find correct columns in input")
            
            
        # create fallacy <--> labelnum mapping
        self.fallacy_list = list(data_df[label_header].unique())
        self.fallacy_dict = {fallacy:i for i, fallacy in enumerate(self.label_to_fallacy)}
        data_df['label'] = data_df[label_header].apply(lambda x: self.fallacy_dict[x])
        data_df = data_df[~data_df.label.astype(int).gt(12)]

        self.data = (data_df[["source_article",label_header,'label']]
                .rename(columns= {"source_article":"text",label_header:"fallacy"})
                .to_dict(orient="records")
            )
        # load embeddings
        
        self.embeds = torch.load(embeds_path).float()
        assert(self.embeds.shape[0] == len(self.data) )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {"label": self.data[idx]["label"], 
                "feats": self.embeds[idx,:]}
        return item    

    @property
    def label_to_fallacy(self):
        return self.fallacy_list

    @property
    def fallacy_to_label(self):
        return self.fallacy_dict
