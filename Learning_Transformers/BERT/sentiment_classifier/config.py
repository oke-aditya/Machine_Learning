import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALIDATION_BATCH_SIZE = 4
EPOCHS = 10
# ACCUMULATION = 2
BERT_PATH = "../models/bert_base_uncased/"
MODEL_PATH = "../models/bert_base_uncased/sentiment_model.bin"
TRAINING_FILE = "../data/imdb_data.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)


