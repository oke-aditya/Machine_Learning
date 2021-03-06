# -*- coding: utf-8 -*-
"""Text_Generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mSUZ5_PojUfYoWEAMPlUGZuBX10Qyfxv

# Using a character based RNN for text generation

The model is character-based. When training started, the model did not know how to spell an English word, or that words were even a unit of text.

The structure of the output resembles a play—blocks of text generally begin with a speaker name, in all capital letters similar to the dataset.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

"""# Reading the data"""

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print("Length of text = %s" %(len(text)))

print(text[:250])

vocab = sorted(set(text))
print("%s unique characters are in the text" %(len(vocab)))

"""# Preprocessing the text

**Vectorize the text**

Before training, we need to map strings to a numerical representation. Create two lookup tables: one mapping characters to numbers, and another for numbers to characters.
"""

# Dictionary mapping characters to ids
char2idx = {}
for i, u in enumerate(vocab):
    char2idx[u] = i
idx2char = np.array(vocab)

text2int = []
for c in text:
    text2int.append(char2idx[c])
text2int = np.array(text2int)

for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))

"""# The prediction task

Given a character, or a sequence of characters, what is the most probable next character? This is the task we're training the model to perform. The input to the model will be a sequence of characters, and we train the model to predict the output—the following character at each time step.

Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment, what is the next character?

## Create training examples and targets

Next divide the text into example sequences. Each input sequence will contain seq_length characters from the text.

For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.

So break the text into chunks of seq_length+1. For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".

To do this first use the tf.data.Dataset.from_tensor_slices function to convert the text vector into a stream of character indices.
"""

seq_length = 100
examples_per_epoch = len(text)
## Create tensorflow data
char_dataset = tf.data.Dataset.from_tensor_slices(text2int)

for i in char_dataset.take(5):
    print(i,idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

"""For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch:"""

def split_input_chunk(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_chunk)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

"""Each index of these vectors are processed as one time step. For the input at time step 0, the model receives the index for "F" and trys to predict the index for "i" as the next character. At the next timestep, it does the same thing but the RNN considers the previous step context in addition to the current input character."""

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

"""# Create data batches"""

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

"""# Build the model"""

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 256

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.GRU(rnn_units, stateful=True, return_sequences=True, recurrent_dropout=0.2))
    model.add(tf.keras.layers.GRU(rnn_units, stateful=True, return_sequences=True, recurrent_dropout=0.2))
    model.add(tf.keras.layers.GRU(rnn_units, stateful=True, return_sequences=True, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

model.summary()

# Dummy run
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

"""To get actual predictions from the model we need to sample from the output distribution, to get actual character indices. This distribution is defined by the logits over the character vocabulary."""

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

sampled_indices

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

"""# Time to train

At this point the problem can be treated as a standard classification problem. Given the previous RNN state, and the input this time step, predict the class of the next character.
"""

def loss(labels, logits):
    return(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

"""# Running Predictions"""

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

"""The prediction loop

The following code block generates the text:

It Starts by choosing a start string, initializing the RNN state and setting the number of characters to generate.

Get the prediction distribution of the next character using the start string and the RNN state.

Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model.

The RNN state returned by the model is fed back into the model so that it now has more context, instead than only one character. After predicting the next character, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted characters.
"""

def generate_text(model, start_string):
    # Evaluating from the learned model
    # number of characters to generate
    num_generate = 1000

    # Vectorize the input string
    input_eval = []
    for s in start_string:
        input_eval.append(char2idx[s])
    
    input_eval = tf.expand_dims(input_eval, 0)

    # Store our generated text results
    text_generated = []

    # Temperature parameter. Lower means more relevant text but less innovation.
    # Higher means suprising text.

    temperature = 0.4

    # here batch size = 1
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # Remove the batch dimenstion
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))

"""# Custom Training Loop"""

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss

epochs = 10
for epoch in range(epochs):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    # initally hidden is None
    hidden = model.reset_states()

    for batch_n, (inp, target) in enumerate(dataset):
        loss = train_step(inp, target)


        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))
    
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

