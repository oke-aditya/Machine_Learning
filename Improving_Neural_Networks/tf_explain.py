# -*- coding: utf-8 -*-
"""tf_explain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kijHPMT3SYeEPh4GvlogXcmPMizIQy0E

# tf-explain 

- Explaining the Neural Network learning

tf-explain offers interpretability methods for Tensorflow 2.0 to ease neural network’s understanding.
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

!pip install tf-explain

import tf_explain

import tensorflow as tf
print(tf.__version__)

import numpy as np

"""# Loading Data"""

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

y_train = tf.keras.utils.to_categorical(y_train, 10, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, 10, dtype='float32')

"""# Building Model"""

img_input = tf.keras.Input((28,28,1))
x = tf.keras.layers.Conv2D(32, (3,3), activation='elu') (img_input)
x = tf.keras.layers.Conv2D(64, (3,3), activation='elu', name='target_layer') (x)
x = tf.keras.layers.MaxPooling2D((2,2)) (x)
x = tf.keras.layers.Dropout(0.3) (x)
x = tf.keras.layers.Flatten() (x)
x = tf.keras.layers.Dense(128, activation='elu') (x)
x = tf.keras.layers.Dropout(0.4) (x)
x = tf.keras.layers.Dense(10, activation='softmax') (x)

model = tf.keras.Model(img_input, x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

"""# Checking for validation Classes"""

# Select a subset of the validation data to examine
# Here, we choose 5 elements with one hot encoded label "0" == [1, 0, 0, .., 0]
validation_class_zero = (np.array([
    el for el, label in zip(x_test, y_test)
    if np.all(np.argmax(label) == 0)
][0:5]), None)

validation_class_four = (np.array([
    el for el, label in zip(x_test, y_test)
    if np.all(np.argmax(label) == 4)
][0:5]), None)

callbacks_l = [tf_explain.callbacks.GradCAMCallback(validation_class_zero, layer_name='target_layer', class_index=0),
tf_explain.callbacks.GradCAMCallback(validation_class_four, layer_name='target_layer', class_index=4),
tf_explain.callbacks.ActivationsVisualizationCallback(validation_class_zero, layers_name=['target_layer']),
tf_explain.callbacks.SmoothGradCallback(validation_class_zero, class_index=0, num_samples=15, noise=1.),
tf_explain.callbacks.IntegratedGradientsCallback(validation_class_zero, class_index=0, n_steps=10),
tf_explain.callbacks.VanillaGradientsCallback(validation_class_zero, class_index=0),
]

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir logs

history = model.fit(x_train, y_train, epochs=10, callbacks=callbacks_l)



"""# Explaining What a network has learnt"""

from tf_explain.core.grad_cam import GradCAM
# from tf_explain.core.gradients_inputs import GradientsInputs
from tf_explain.core.activations import ExtractActivations
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.integrated_gradients import IntegratedGradients
from tf_explain.core.smoothgrad import SmoothGrad
# from tf_explain.core.vanilla_gradients import VanillaGradients

"""## GradCAM"""

model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

img = tf.keras.preprocessing.image.load_img('/content/cat.jpg', target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)

print(img.shape)

data = (np.array([img]), None)
# print(data.shape)

cat_class_index = 281                  # As in imagenet
explainer = GradCAM()

grid = explainer.explain(data, model, class_index=cat_class_index, layer_name="block5_conv3")
explainer.save(grid, ".", "grad_cam.png")

"""## Activation Maps"""

img = tf.keras.preprocessing.image.load_img('/content/cat.jpg', target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)
print(img.shape)
data = (np.array([img]), None)

target_layers = ["conv1_relu"]
explainer = ExtractActivations()
grid = explainer.explain(data, model, target_layers)    
explaner.save(grid, '.', "activation.png")

"""## Occlusion Sensitivity"""

img = tf.keras.preprocessing.image.load_img('/content/cat.jpg', target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)
print(img.shape)
data = (np.array([img]), None)

cat_class_index = 281
explainer = OcclusionSensitivity()

grid = explainer.explain(data, model, cat_class_index, patch_size=20)
explainer.save(grid, ".", "occlusion_analysis20.png")

grid = explainer.explain(data, model, cat_class_index, patch_size=10)
explainer.save(grid, ".", "occlusion_analysis10.png")

"""## Integrated Gradients"""

img = tf.keras.preprocessing.image.load_img('/content/cat.jpg', target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)
print(img.shape)
data = (np.array([img]), None)

cat_class_index = 281
explainer = IntegratedGradients()

grid = explainer.explain(data, model, cat_class_index, n_steps=15)
explainer.save(grid, ".", "integrated_gradients.png")

"""## Smooth Grad"""

img = tf.keras.preprocessing.image.load_img('/content/cat.jpg', target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)
print(img.shape)
data = (np.array([img]), None)

cat_class_index = 281
explainer = SmoothGrad()

grid = explainer.explain(data, model, cat_class_index, num_samples=20, noise=1.0)
explainer.save(grid, ".", "smooth_gradients.png")