# -*- coding: utf-8 -*-
"""keras_tuner.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FbSUg_-KxlihGbwBPBEBLFzCw-YtVLhE

# Keras Tuner

- A package to tune Keras hyperparameters
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

!pip install keras-tuner

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

"""Label   Description
- 0   T-shirt/top
- 1   Trouser
- 2   Pullover
- 3   Dress
- 4   Coat
- 5   Sandal
- 6   Shirt
- 7   Sneaker
- 8   Bag
- 9   Ankle boot
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

import time
import pickle

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

"""# Loading Data"""

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

print(x_train.shape)
print(x_test.shape)

plt.imshow(x_train[1], cmap="gray")
plt.show()

"""# Non-keras Tuner Model"""

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu',))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

model = build_model()

model.summary()

history = model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, validation_split=0.2, callbacks=None)

"""# With Keras-Tuner Model"""

def build_model_tuned(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Int("input_units", min_value=32, max_value=256, step=32), (3,3), input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
    
    for i in range(hp.Int('n_layers',1,4)):
        model.add(tf.keras.layers.Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3,3)))
        model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

LOG_DIR = f"{int(time.time())}"

"""# Metrics and Performance"""

tuner = RandomSearch(build_model_tuned, objective="val_acc", 
                     max_trials = 1,
                     executions_per_trial = 1,
                     directory = LOG_DIR)

tuner.search(x=x_train, y=y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

tuner.search_space_summary()

tuner.results_summary()

print(tuner.get_best_hyperparameters()[0].values)

best_model = tuner.get_best_models()[0]

best_model.summary()

"""# In Short what we did"""

optimizer = hp.Choice('optimizer', ['adam', 'sgd'])

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

def build_tuned_model2(hp, num_classes=10):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), (3,3), input_shape = x_train.shape[1:]))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    for i in range(hp.Int('n_layers',1, 4)):
        model.add(tf.keras.layers.Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3,3)))
        model.add(tf.keras.layers.Activation('elu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='elu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])

    return model

tuner = RandomSearch(build_tuned_model2, objective='val_acc', max_trials=20, executions_per_trial=2, directory=LOG_DIR)

tuner.search_space_summary()

tuner.search(x=x_train, y=y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

"""# Loading and Saving the Model using Pickle"""

with open(f"tuner_{int(time.time())}.pkl", "wb") as f:
    pickle.dump(tuner,f)

tuner = pickle.load(open("/content/tuner_1577190691.pkl","rb"))
tuner.get_best_hyperparameters()[0].values

best_model2 = tuner.get_best_models()[0]

best_model2.summary()

"""# Some extra Parameters to tune

- Use hp.Choice to provide choice to the hyperparametrs
"""

def build_model_add(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=hp.Int('input_units', min_value=32, max_value=512, step=32),kernel_size=(3,3),
                                     activation='relu', input_shape=x_train.shape[1:]))
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # To customize the learning rate

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-4, 1e-5, 1e-6])), 
                  loss="sparse_categorical_crossentropy", metrics=['acc'])
    
    return model

tuned_model3 = RandomSearch(
    build_model_add,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

"""# SubClassing to build a new HyperTuner Model"""

from kerastuner import HyperModel

class MyHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def build(self, hp):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='elu'))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-4,1e-5,1e-6])),
                      loss='categorical_crossentropy', metrics=['acc'])
        
        return model

hypermodel = MyHyperModel(num_classes=10)

tuner = RandomSearch(hypermodel, objective='val_acc', max_trials=10, directory='LOG_DIR', project_name='new_project1')

tuner.search(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test))

"""# Using Premade Tunable Applications Hyper-Resnet and HyperXception

- These are ready-to-use hypermodels for computer vision.

- They come pre-compiled with loss="categorical_crossentropy" and metrics=["accuracy"].
"""

from kerastuner.applications import HyperResNet
from kerastuner import HyperParameters
from kerastuner.tuners import Hyperband

hypermodel = HyperResNet(input_shape=(28,28,1), classes=10)

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=40,
    directory='my_dir',
    project_name='helloworld')

tuner.search(x_train, y_train,
             epochs=20,
             validation_data=(x_test, y_test))

"""# Restricting the Search space"""

hypermodel = HyperXception(input_shape=(28, 28, 1), classes=10)
hp = HyperParameters()

hp.Choice('learning_rate', values=[1e-4,1e-5,1e-6])

tuner = Hyperband(hypermodel, hyperparameters=hp, tune_new_entries=False, objective='val_acc', max_epochs=40, directory='my_dir',
                  project_name='try3')

tuner.search(x_train, y_train, epochs=20,  validation_data=(x_test, y_test))

"""# Over-riding the Compilation Arguments"""

hypermodel = HyperXception(input_shape=(28, 28, 1), classes=10)

tuner = Hyperband(hypermodel, optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse',
                  metrics=[tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')], objective='val_acc',
                  max_epochs=40, directory='my_dr', project_name="hello")

tuner.search(x_train, y_train, epochs=20, validation_data=(x_test,y_test))

