import tensorflow_datasets as tfds
import tensorflow as tf
from flask import Flask, jsonify, make_response, request
from healthcheck import HealthCheck
import logging

app = Flask(__name__)
padding_size = 1000
model = tf.keras.models.load_model('sentiment_analysis.hdf5')
text_encoder = tfds.features.text.TokenTextEncoder.load_from_file('sa_encoder.vocab')
print("--------------Model and vocabulary loaded--------------")

###

logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s")

logging.info('Model and Vocabulary Loaded')  

# This is not recommended though because it will not go to correct app.

health = HealthCheck(app, "/hcheck") # If we call url /hcheck then we get health of app

# To make sure our app is up and running
def isup():
    return True, "App is up"

health.add_check(isup)

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def predict_fn(pred_text, pad_size):
    encoded_pred_text = text_encoder.encode(pred_text)
    encoded_pred_text = pad_to_size(encoded_pred_text, pad_size)
    encoded_pred_text = tf.cast(encoded_pred_text, tf.int64)
    predictions = model.predict(tf.expand_dims(encoded_pred_text, 0)) # To add the batch size
    return predictions


@app.route('/setclassifier', methods=['POST'])
def predict_sentiment():
    text = request.get_json()['text']
    print(text)
    print(type(text))
    predictions = predict_fn(text, padding_size)
    if(float(predictions[0][0]) > 0.0):
        sentiment = 'positive'
    else:
        sentiment = 'negative'
    
    # This is the correct method.
    app.logger.info("Prediction : " + str(predictions[0][0]) + "sentiment :" + sentiment)
    return jsonify({'sentiment' : sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)

