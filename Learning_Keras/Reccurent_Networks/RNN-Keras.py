# It takes input of shapes (batch_size, timestamp, input_features)
# It has two output modes it can return either the full sequence of successive outputs as 3D tensor
# (batch_size, timestamp, output_features)
# Or last output can be of the shape 2D tensor
# (batch_size, features)
# These two modes are controlled by the return sequences argument

from keras.models import Sequential
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 1000
maxlen = 500
batch_size = 32
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
print("Train sequences",len(x_train))
print("Test sequences",len(x_test))

print(x_train.shape)
print(x_test.shape)
# print(x_train)

input_train = sequence.pad_sequences(x_train, maxlen = maxlen)
input_test = sequence.pad_sequences(x_test, maxlen = maxlen)

print("Input shape", input_train.shape)
print("Input labels",y_train.shape)
print("Input test", input_test.shape)
print("Test labels",y_test.shape)
# print(input_train)

model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'Adam', metrics = ['acc'], loss = 'binary_crossentropy')
history = model.fit(input_train, y_train, epochs = 10, verbose = -1, validation_split=0.2, batch_size=32)



