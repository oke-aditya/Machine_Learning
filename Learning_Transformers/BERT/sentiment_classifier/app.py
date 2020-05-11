import config
# import dataset
import torch

from flask import Flask, jsonify
from flask import request
from model import BERTBasedUncased

app = Flask(__name__)

MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sentence_prediction(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    
    inputs = tokenizer.encode_plus(
            review, 
            None,
            add_special_tokens=True,
            max_length = max_len
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    # BERT We need to padd on right side.
    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    
    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)


    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    print(sentence)
    positive_prediction = sentence_prediction(sentence, model=MODEL)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive_prediction" : str(positive_prediction),
        "negative_prediction" : str(negative_prediction),
        "sentence" : str(sentence)
    }

    return jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBasedUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)
