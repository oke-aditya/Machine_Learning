# -*- coding: utf-8 -*-
"""Tensorflow_Hub.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sw73odd0KJi5icmuWxYtvlL-LidiN7t3

# Image Classification Using TensorFlow Hub

In this chapter, we will cover the following topics:

1. Getting the data
2. Transfer learning
3. Fine-tuning

# Getting the Data

The task we are going to solve in this chapter is a classification problem on a dataset of flowers, which is available in tensorflow-datasets (tfds). The dataset's name is tf_flowers and it consists of images of five different flower species at different resolutions.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
print(tf.__version__)

from time import time

import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", with_info=True)
print(info)

"""There is a single split train with 3,670 labeled images. The image resolution is not fixed, as we can see from the None value in the height and width position of the Image shape feature. There are five classes, as expected. 

Classes = Daisy Dandelion Roses Sunflowers Tulips
"""

dataset = dataset["train"]
tot = 3670

train_set_size = tot // 2
validation_set_size = tot - train_set_size - train_set_size // 2
test_set_size = train_set_size - validation_set_size

print("train set size: ", train_set_size)
print("validation set size: ", validation_set_size)
print("test set size: ", test_set_size)

train, test, validation = (dataset.take(train_set_size),
                           dataset.skip(train_set_size).take(validation_set_size),
                           dataset.skip(train_set_size + validation_set_size).take(test_set_size))

"""# Transfer Learning

Transfer learning is the process of learning a new task by relying on a previously learned task: the learning process can be faster, more accurate, and require less training data.

Transferring the knowledge of a trained model to a new one requires us to remove the task-specific part of the network (which is the classification layers) and keep the CNN fixed as the feature extractor.

# Tensorflow Hub

TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a self-contained piece of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning. Transfer learning can:

- Train a model with a smaller dataset
- Improve generalization, and
- Speed up training
"""

!pip install tensorflow-hub>0.3

"""The TensorFlow 2.0 integration is terrific—we only need the URL of the module on TensorFlow Hub to create a Keras layer that contains the parts of the model we need!

There are models in both versions: feature vector-only and classification, which means a feature vector plus the trained classification head. The TensorFlow Hub catalog already contains everything we need for transfer learning
"""

def to_float_image(example):
    example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
    return example

def resize(example):
    example["image"] = tf.image.resize(example["image"], (299, 299))
    return example

train = train.map(to_float_image).map(resize)
validation = validation.map(to_float_image).map(resize)
test = test.map(to_float_image).map(resize)

print(test)

"""The TensorFlow Hub Python package has already been installed, and this is all we need to do:

- Download the model parameters and graph description
- Restore the parameters in its graph
- Create a Keras layer that wraps the graph and allows us to use it like any other Keras layer we are used to using
"""

import tensorflow_hub as hub

hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2", output_shape=[2048], trainable=False)

"""The model definition is straightforward, thanks to the Keras integration. Everything is set up to define the training loop, measure the performance, and see whether the transfer learning approach gives us the expected classification results."""

num_classes = 5
model = tf.keras.Sequential()
model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2", output_shape=[2048], trainable=False))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes))

# model.summary()

"""To enable a progress bar, using the TFHUB_DOWNLOAD_PROGRESS environment variable is required by hub.KerasLayer. Therefore, on top of the script, the following snippet can be added,"""

import os
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"

"""# Training and evaluating

Using a pre-trained feature extractor allows us to speed up the training while keeping the training loop, the losses, and optimizers unchanged, using the same structure of every standard classifier train.

Since the dataset labels are tf.int64 scalars, the loss that is going to be used is the standard sparse categorical cross-entropy, setting the from_logits parameter to True.
"""

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
step = tf.Variable(1, name="global_step", trainable=False)
optimizer = tf.optimizers.Adam(1e-3)

# train_summary_writer = tf.summary.create_file_writer("./log/train_summary")
# val_summary_writer = tf.summary.create_file_writer('./log/val_summary')

accuracy = tf.metrics.Accuracy()
metrics = tf.metrics.Mean(name="loss")

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss(labels, logits)   
    
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    step.assign_add(1)

    accuracy.update_state(labels, tf.argmax(logits, -1))
    return loss_value


train = train.batch(32).prefetch(1)
validation = validation.batch(32).prefetch(1)
test = test.batch(32).prefetch(1)

num_epochs = 10
for epoch in range(num_epochs):
    start = time()
    for example in train:
        image, label = example["image"], example["label"]
        loss_value = train_step(image, label)
        mean_loss.update_state(loss_value)

        if tf.equal(tf.math.mod(step, 10), 0):
            tf.print("Step no: %d \n Loss: %d \n accuracy: %0.2f" %(step, mean_loss.result(), accuracy.result()))
            mean_loss.reset_states()
            accuracy.reset_states()
        
    end = time()
    print("Time per epoch: %0.2f"%(epoch))
    # End of an epoch, time to validate
    tf.print("######### Validation at epoch %d ###########" %(epoch))
    accuracy.reset_states()
    for example in validation:
        image, label = example["image"], example["label"]
        logits = model(image)
        accuracy.update_state(label, tf.argmax(logits, -1))

    tf.print("Validation accuracy: %0.2f" %(accuracy.resul()))
    accuracy.reset_states()

"""# Fine-tuning


- Fine-tuning is a different approach to transfer learning. Both share the same goal of transferring the knowledge learned on a dataset on a specific task to a different dataset and a different task.

- Fine-tuning, instead, consists of fine-tuning the pre-trained network weights by continuing backpropagation.

Different points to keep in mind when thinking about fine-tuning:

- **Dataset size:** Fine-tuning a network means using a network with a lot of trainable parameters, and, as we know from the previous chapters, a network with a lot of parameters is prone to overfitting.
If the target dataset size is small, it is not a good idea to fine-tune the network. Using the network as a fixed-feature extractor will probably bring in better results.

- **Dataset similarity:** If the dataset size is large (where large means with a size comparable to the one the pre-trained model has been trained on) and it is similar to the original one, fine-tuning the model is probably a good idea. Slightly adjusting the network parameters will help the network to specialize in the extraction of features that are specific to this dataset, while correctly reusing the knowledge from the previous, similar dataset.
If the dataset size is large and it is very different from the original, fine-tuning the network could help. In fact, the initial solution of the optimization problem is likely to be close to a good minimum when starting with a pre-trained model, even if the dataset has different features to learn (this is because the lower layers of the CNN usually learn low-level features that are common to every classification task).

Using a high learning rate would change the network parameters too much, and we don't want to change them in this way. Instead, using a small learning rate, we slightly adjust the parameters to make them adapt to the new dataset, without distorting them too much, thus reusing the knowledge without destroying it.

# Fine Tuning using Tensorflow Hub

1. Download the model parameters and graph
2. Restore the model parameters in the graph
3. Restore all the operations that are executed only during the training (activating dropout layers and enabling the moving mean and variance computed by the batch normalization layers)
4. Attach the new layers on top of the feature vector
5. Train the model end to end
"""

hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",output_shape=[2048],trainable=True)

"""## Train and evaluate"""

optimizer = tf.keras.optimizers.Adam(1e-5)
model = tf.keras.Sequential()
model.add(hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",output_shape=[2048],trainable=True))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# SAME TRAINING LOOP
# Very slow training speed.