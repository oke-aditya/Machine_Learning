# -*- coding: utf-8 -*-
"""Model_deployment_tf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_w60NzMF9Iiv3IMSvgkCqPdLkGGkXrbM


# Deployment Code

## Load model and check stuff
"""

# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf

print(tf.__version__)

MODEL_PATH = '/content/drive/My Drive/models/sentiment_analysis.hdf5'
model = tf.keras.models.load_model(MODEL_PATH)
print(model.outputs)

print(model.inputs)

model.summary()

model.get_weights()

emb_layer = model.get_layer(index=0)

emb_weights = emb_layer.get_weights()[0]

emb_weights.shape

lstm_layer = model.get_layer(index=1)

lstm_layer.output_shape

lstm_layer.weights

"""## Deploying"""

import tensorflow_datasets as tfds
import tensorflow as tf

model = tf.keras.models.load_model('//content//drive//My Drive//models//sentiment_analysis.hdf5')

text_encoder = tfds.features.text.TokenTextEncoder.load_from_file('/content/sa_encoder.vocab')

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

"""# Checking the pipeline"""

pred_text = ('Worst prouct. Never buy such cheap stuff. Seller gave very late.')
predictions = predict_fn(pred_text, 1000)
print(predictions)

pred_text = ('Beautiful product. Masterpiece. I would recommend this to anyone and everyone. Great delivery')
predictions = predict_fn(pred_text, 1000)
print(predictions)