# # 1-D Convonets


## Introduction

# - 1-D convonents can be used for text data.
# - They are faster than RNNs and at times more useful for audio / signal processing.

# - Such 1D convolution layers can recognize local patterns in a sequence. Because the
# same input transformation is performed on every patch, a pattern learned at a certain
# position in a sentence can later be recognized at a different position, making 1D convnets
# translation invariant (for temporal translations). 

# - For instance, a 1D convnet processing
# sequences of characters using convolution windows of size 5 should be able to

# learn words or word fragments of length 5 or less


# - The 2D poolingm operation has a 1D equivalent: extracting 1D patches (subsequences) from an input
# and outputting the maximum value (max pooling) or average value (average pooling).
# - Just as with 2D convnets, this is used for reducing the length of 1D inputs (subsampling).


# ## Working with IMDB

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras import optimizers


max_features = 1000       # Consider only top 1000 words as features
max_len = 500             # Maximum length of review 500



(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)



print("Number of reviews in train = %d" %(len(x_train)))
print("Number of reviews in test = %d" %(len(x_test)))



x_padded_train = sequence.pad_sequences(x_train, maxlen = max_len)
x_padded_test = sequence.pad_sequences(x_test, maxlen= max_len)


# ## Tips for using 1D Convonets

# - 1D convnets are structured in the same way as their 2D counterparts, which you used
# in chapter 5: they consist of a stack of Conv1D and MaxPooling1D layers, ending in
# either a global pooling layer or a Flatten layer, that turn the 3D outputs into 2D outputs,
# allowing you to add one or more Dense layers to the model for classification or
# regression.
# - One difference, though, is the fact that you can afford to use larger convolution
# windows with 1D convnets. 
# - With a 2D convolution layer, a 3 × 3 convolution window
# contains 3 × 3 = 9 feature vectors; but with a 1D convolution layer, a convolution window
# of size 3 contains only 3 feature vectors. 
# - You can thus easily afford 1D convolution windows of size 7 or 9.


model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, kernel_size=7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, kernel_size=7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))



opt = optimizers.Adam(lr = 0.001, decay=1e-5)
model.compile(optimizer =  opt, loss = 'binary_crossentropy', metrics=['acc'])



model.summary()



history = model.fit(x_padded_train, y_train, validation_data=(x_padded_test, y_test) ,epochs = 2, batch_size=32, verbose = 1)



