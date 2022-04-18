import torch.nn as nn 
import torch.nn.functional as F 
from torch import sigmoid
from copy import deepcopy 


class FallacyModel(nn.Module): 
    
    def __init__(self,n_hiddens=50,n_classes=13): 
        super().__init__()
        
        self.dropout = nn.Dropout(p=0.5) 
        
        self.lin_1 = nn.Linear(768, n_hiddens)
        self.lin_2 = nn.Linear(n_hiddens,n_classes) 
        
    def forward(self, x): 
        x = self.dropout(x)
        x = F.relu(self.lin_1(x))
        x = self.dropout(x)
        z = self.lin_2(x)
        return z
    
class ProppyModel(nn.Module):
    def __init__(self,from_model=None, n_hiddens=50,dropout_p = 0.5): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p) 
        if from_model is not None: 
            n_hiddens = from_model.lin_1.out_features
            self.lin_1 = deepcopy(from_model.lin_1)
        else: 
            self.lin_1 = nn.Linear(768, n_hiddens)
            
        self.lin_2 = nn.Linear(n_hiddens,2) 
        
    def forward(self, x): 
        x = self.dropout(x)
        x = F.relu(self.lin_1(x))
        x = self.dropout(x)
        z = self.lin_2(x)
        return z
