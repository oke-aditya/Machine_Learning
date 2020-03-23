# -*- coding: utf-8 -*-
"""Text_classification_RNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yV-iJxHIIvwg87bNk_jKagd_yCDbMQWT

# Padding and Masking

Padding

- Adding zeros at end of the sequence to keep the sequence length constant
"""

text = [
  ["The", "weather", "will", "be", "nice", "tomorrow"],
  ["How", "are", "you", "doing", "today"],
  ["Hello", "world", "!"]
]
tokenzing = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [71, 1331, 4231]
]

padding = [[  83   91    1  645 1253  927]
 [  73    8 3215   55  927    0]
 [ 711  632   71    0    0    0]]

"""Masking

- "Masking" is how layers are able to know when to skip / ignore certain timesteps in sequence inputs.
- Some layers are mask-generators: Embedding can generate a mask from input values (if mask_zero=True), and so can the Masking layer.
- Some layers are mask-consumers: they expose a mask argument in their __call__ method. This is the case for RNN layers.
- In the Functional API and Sequential API, mask information is propagated automatically.
- When writing subclassed models or when using layers in a standalone way, pass the mask arguments to layers manually.
0 You can easily write layers that modify the current mask, that generate a new mask, or that consume the mask associated with the inputs.

# Test classification
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

"""# Load IMDB Data

- IMDB data is already tokenzied so we dont need to tokenize it
"""

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64

"""- Note: As of TensorFlow 2.2 the padded_shapes argument is no longer required. The default behavior is to pad all axes to the longest in the batch.

In tf v2.2 use

train_dataset = (train_examples
                 .shuffle(BUFFER_SIZE)
                 .padded_batch(BATCH_SIZE))

test_dataset = (test_examples
                .padded_batch(BATCH_SIZE))
"""

train_dataset = (train_examples.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

test_dataset = (test_examples.padded_batch(BATCH_SIZE, padded_shapes=([None], [])))

"""# Build the model"""

encoder = info.features['text'].encoder

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(encoder.vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True, recurrent_dropout=0.2)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True, recurrent_dropout=0.2)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

model.summary()

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

"""The above model does not mask the padding applied to the sequences. This can lead to skew if trained on padded sequences and test on un-padded sequences. Ideally you would use masking to avoid this, but as you can see below it only have a small effect on the output."""
