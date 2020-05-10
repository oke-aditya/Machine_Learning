import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert_path = bert_path
        

