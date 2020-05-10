import torch.nn as nn
import torch
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import model_selection
import numpy as np
import pandas as pd

class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)
    
    def forward(self, ids, mask, token_type_ids):
        # We do not need the sequential output
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # Pool output has 768 features hence the Linear layer has 768 items.
        bo = self.bert_drop(o2)
        return self.out(bo)

class BERTDatasetTraining:
    def __init__(self, qtitle, qbody, answer, targets, tokenizer, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = targets
    
    def __len__(self):
        return len(self.answer)
    
    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer = str(self.answer[item])

        # [CLS] [Q-TITLE] [Q-BODY] [SEP] [ANSWER] [SEP]

        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body,
            answer,
            add_special_tokens=True,
            max_len = self.max_len
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        token_type_ids = token_type_ids  + ([0] * padding_len)
        mask = mask + ([0] * padding_len)


        return {
            "ids" : torch.Tensor(ids, dtype=torch.long),
            "mask" : torch.Tensor(mask, dtype=torch.long),
            "token_type_ids" : torch.Tensor(token_type_ids, dtype=torch.long),
            "targets" : torch.Tensor(self.targets[item, :], dtype=torch.float)
        }

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
        	scheduler.step()


def eval_loop_fn(data_loader, model, device):
	model.eval()
    fin_targets = []
    fin_outputs = []

    for bi, d in enumerate(data_loader):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)

        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())


    return np.vstack(fin_outputs), np.vstack(fin_targets)


def run():
	MAX_LEN = 512
	TRAIN_BATCH_SIZE = 4
	EPOCHS = 20

	dfx = pd.read_csv("../input/train.csv").fillna("none")
	df_train, df_valid = model_selection.train_test_split(dfx, random_state=31, test_size=0.1, )
	df_train.reset_index(drop=True)
	df_val.reset_index(drop=True)

	sample = pd.read_csv('..input/sample_submission.csv')
	target_cols = list(sample.drop("qa_id", axis=1).columns)
	train_targets = df_train[target_cols].values
	valid_targets = df_val[target_cols].values

	tokenizer = transformers.BertTokenizer.from_pretrained("..input/bert_base_uncased/")

	train_dataset = BERTDatasetTraining(qtitle=df_train.question_title.values,
		qbody=df_train.question_body.values,
		answer=df_train.answer.values,
		targets=train_targets,
		tokenizer=tokenizer,
		max_len=MAX_LEN)

	train_data_loader = torch.utils.DataLoader(
		train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


	valid_dataset = BERTDatasetTraining(qtitle=df_valid.question_title.values,
		qbody=df_valid.question_body.values,
		answer=df_valid.answer.values,
		targets=valid_targets,
		tokenizer=tokenizer,
		max_len=MAX_LEN)

	valid_data_loader = torch.utils.DataLoader(
		valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

	# For GPU
	device = "cuda"
	model = BERTBaseUncased("../input/bert_base_uncased").to(device)
	lr = 3e-5
	num_training_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE * EPOCHS)
	optimizer = AdamW(model.parameters, lr=lr)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

	for epoch in range(EPOCHS):
		train_loop_fn(train_data_loader, model, optimizer, device, scheduler)
		o, t = eval_loop_fn(valid_data_loader, model, device)

		spear = []
		for jj in range(t.shape[1]):
			p1 = list(t[:, jj])
			p2 = list(t[:, jj])
			coef, _ = np.nan_to_num(stats.spearman(p1, p2))
			spear.append(coef)

		spear = np.mean(spear)
		print(f" epoch = {epoch} pearman = {spear}")

 
if __name__ == '__main__':
	run()







