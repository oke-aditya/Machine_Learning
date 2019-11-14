
# # Bidirectional Recurrent Neural Networks

# ## Introduction

# - A bidirectional RNN is a common RNN variant that can offer greater performance than a regular
# RNN on certain tasks. 
# - It’s frequently used in natural-language processing
# - You could call it the Swiss Army knife of deep learning for natural-language processing.

# - RNNs are notably order dependent, or time dependent: they process the timesteps
# of their input sequences in order, and shuffling or reversing the timesteps can completely
# change the representations the RNN extracts from the sequence. 
# - This is precisely the reason they perform well on problems where order is meaningful, such as
# the temperature-forecasting problem. 
# - A bidirectional RNN exploits the order sensitivity of RNNs:
# - it consists of using two regular RNNs, such as the GRU and LSTM layers  each of which processes the input sequence in one direction (chronologically and antichronologically), and then merging their representations.
# - By processing a sequence both ways, a bidirectional RNN can catch patterns that may be overlooked by a unidirectional RNN

# ## Reversed Order GRU


from keras import layers
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential



max_features = 1000
maxlen = 50



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)



x_padded_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_padded_test = sequence.pad_sequences(x_test, maxlen=maxlen)



x_rev_train =[x[::-1] for x in x_train]
x_rev_test = [x[::-1] for x in x_test]



x_rev_train_padded = sequence.pad_sequences(x_rev_train, maxlen=maxlen)
x_rev_test_padded = sequence.pad_sequences(x_rev_test, maxlen=maxlen)



model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.GRU(32, activation='tanh', recurrent_dropout=0.3, dropout = 0.1))
model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])



model.fit(x_rev_train_padded, y_train, validation_data=(x_rev_test_padded, y_test), batch_size=64, epochs=2, verbose=1)


# - A reverse order GRU works as good as a chrnonological order GRU.
# - We can expect this as it does not matter for a task such as sentiment classification. 
# - But the features learnt by the GRU are completely different than that of a chrnonological GRU.
# - We can exploit this idea.


# ## Why bidirectional RNNs


# - A bidirectional RNN exploits this idea to improve on the performance of chronologicalorder
# RNNs. 
# - It looks at its input sequence both ways (see figure 6.25), obtaining potentially
# richer representations and capturing patterns that may have been missed by the
# chronological-order version alone.


model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.GRU(32, activation='tanh', dropout=0.3, recurrent_dropout=0.2)))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'])
history = model.fit(x_padded_train, y_train, validation_data=(x_padded_test, y_test), verbose = 1, epochs=2, batch_size=64) 




