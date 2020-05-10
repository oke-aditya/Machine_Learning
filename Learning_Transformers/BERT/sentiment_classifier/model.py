# Fine tune the BERT model
import transformers
import config
import torch.nn as nn

class BERTBasedUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        # 1 for binary classification
    
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # Vector of size 768 for each sample in batch.
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

