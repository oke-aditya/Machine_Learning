import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        # we have tokenizer and remaining stuff in config
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, item):
        # returns items of our dataset
        review = str(self.review)
        review = " ".join(review.split()) # Remove spaces

        # Tokenizer usually encodes 2 strings at a time. We don't have the second string.
        inputs = self.tokenizer.encode_plus(
            review, 
            None,
            add_special_tokens=True,
            max_length = self.max_len
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # BERT We need to padd on right side.
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            "ids" : torch.tensor(ids, dtype=torch.long),
            "mask" : torch.tensor(mask, dtype=torch.long),
            "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
            "target" : torch.tensor(self.target[item], dtype=torch.float) 
        }


        