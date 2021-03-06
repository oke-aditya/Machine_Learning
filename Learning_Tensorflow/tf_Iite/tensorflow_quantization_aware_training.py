# -*- coding: utf-8 -*-
"""TensorFlow_Quantization_Aware_Training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sp3nQMPnkJWl7HOBKhOR3LSxNM3wF7FN

# Quantization Aware Training Using Tensorflow

- This creates a model which is already quantized.
- This is opposite to post model quantization where we prune the weights of the model after training.
- Here we train a model that already knows what quantized weights are.
- It does reduced precision training.
- Advantage over post model is accuracy. Model is qunatization aware so it is better than post quantization.
- It does integer arithmetic, so it is pretty fast with CPU.
- Train on GPU infer of CPU.
- Useful for edge Devices.

# GET THE DATA
"""

!nvidia-smi

!pip install -q tensorflow-model-optimization

import tensorflow as tf
print(tf.__version__)
import os
import tensorflow_datasets as tfds
import datetime

tf.config.list_physical_devices()

dataset, info = tfds.load(name="fashion_mnist", with_info=True, as_supervised=True, try_gcs=True, split=["train", "test"])

print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

fm_train, fm_test = dataset[0], dataset[1]
fm_val = fm_test.take(30000)
fm_test = fm_test.skip(30000).take(30000)

# print(len(list(fm_train)))

"""# Inspect Data"""

import matplotlib.pyplot as plt
import numpy as np

for fm_sample in fm_train.take(5):
    image, label = fm_sample[0], fm_sample[1]

    plt.figure()
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap="gray") 
    print("Label %d " %label.numpy())
    print("Category %s "%info.features["label"].names[label.numpy()])
    plt.show()

"""# Get THE MODEL"""

def scale(img, label):
    img = tf.cast(img, tf.float32)
    img /= 255.
    return img, label

def get_dataset(batch_size=32):
    train_dataset_scaled = fm_train.map(scale).shuffle(60000).batch(batch_size)
    test_dataset_scaled = fm_test.map(scale).batch(batch_size)
    val_dataset_scaled = fm_val.map(scale).batch(batch_size)

    return train_dataset_scaled, test_dataset_scaled, val_dataset_scaled

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, 2, padding="same", activation="relu", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv2D(128, 2, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, 2, padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

"""# Quantize The Model"""

import tensorflow_model_optimization as tfmot

model = create_model()

quantized_model = tfmot.quantization.keras.quantize_model(model)

quantized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

quantized_model.summary()

logdir = os.path.join("/tmp/logs", datetime.datetime.now().strftime("%Y%m%d-%H%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

train_dataset, test_dataset, val_dataset = get_dataset()

train_dataset.cache()
val_dataset.cache()

history = quantized_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=[tensorboard_callback])

"""- Still after training the output is float 32 weights.

- It is just aware of Quantization. We still need to quantize it so that it will be in int.
"""

model.save("/tmp/fashion_model.h5")

"""# Check the tensorBoard"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir /tmp/logs

!nvidia-smi

print(quantized_model)

quantized_model.evaluate(train_dataset)

quantized_model.evaluate(val_dataset)

"""# Quantize it Fully"""

converter = tf.lite.TFLiteConverter.from_keras_model(quantized_model)
converter_optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

quantized_model_size = len(quantized_tflite_model) / 1024
print("Size of quantized model %d KBs"%(quantized_model_size))

"""# Infer on the Qunatized Model"""

interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

input_tensor_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.tensor(interpreter.get_output_details()[0]["index"])

interpreter.get_tensor_details()

prediction_output = []
accurate_count = 0

for test_image in fm_val.map(scale):
    # print("Hi wWT")
    test_image_p = np.expand_dims(test_image[0], axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image_p)

    interpreter.invoke()
    out = np.argmax(output_index()[0])
    prediction_output.append(out)
    # print(out)

    if out == int(test_image[1].numpy()):
        accurate_count += 1

accuracy = accurate_count / len(prediction_output)

print("Accuracy = %0.4f "%(accuracy * 100))