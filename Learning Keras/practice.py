from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, SimpleRNN, Embedding, LSTM
from keras.models import Sequential

max_features = 10000            # Consider number of words
maxlen  = 5000                  # Consider first 500 words of review
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_padded_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_padded_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_padded_train, y_train, epochs=2, batch_size=batch_size, verbose=1, validation_data=(x_padded_test,y_test))



