import tensorflow_datasets as tfds
import tensorflow as tf

from flask import Flask, jsonify, make_response, request

app = Flask(__name__)
padding_size = 1000

model = tf.keras.models.load_model('models\sentiment_analysis.hdf5')
text_encoder = tfds.features.text.TokenTextEncoder.load_from_file('models\sa_encoder.vocab')

print("--------------Model and vocabulary loaded--------------")

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def predict_fn(pred_text, pad_size):
    encoded_pred_text = text_encoder.encode(pred_text)
    encoded_pred_text = pad_to_size(encoded_pred_text, pad_size)
    encoded_pred_text = tf.cast(encoded_pred_text, tf.int64)
    predictions = model.predict(tf.expand_dims(encoded_pred_text, 0)) # To add the batch size

    return predictions.to_list()


