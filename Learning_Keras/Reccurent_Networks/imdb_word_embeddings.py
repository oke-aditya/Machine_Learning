import os
from keras.preprocessing.text import Tokenizer
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

imdb_dir = "E:\Aditya\AI_DL_ML\Datasets\imdb"
print(imdb_dir)
train_dir = os.path.join(imdb_dir, "train")
test_dir = os.path.join(imdb_dir, "test")
print(train_dir) 
print(test_dir)
labels = []
texts = []
test_texts = []
test_labels = []

print("Reading from the train directory ")
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(train_dir, label_type)
    print(dir_name)
##    print(dir_name)
    for filename in tqdm(os.listdir(dir_name)):
##        print(filename)
        if(filename[-4:] == '.txt'):
            f = open(os.path.join(dir_name, filename))
            try:
                texts.append(f.read())
                f.close()
                if(label_type == 'neg'):
                   labels.append(0)
                else:
                   labels.append(1)
            except UnicodeDecodeError:
                f.close()
                continue

print("Reading from the test directory ")
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(test_dir, label_type)
    # print(dir_name)
##    print(dir_name)
    for filename in tqdm(os.listdir(dir_name)):
##        print(filename)
        if(filename[-4:] == '.txt'):
            f = open(os.path.join(dir_name, filename))
            try:
                test_texts.append(f.read())
                f.close()
                if(label_type == 'neg'):
                   test_labels.append(0)
                else:
                   test_labels.append(1)
            except UnicodeDecodeError:
                f.close()
                continue

            
print("Read %d number of train reiews" %(len(labels)))
print("Read %d number of train texts" %(len(texts)))

print("Read %d number of test reviews" %(len(test_labels)))
print("Read %d number of test texts" %(len(test_texts)))

maxlen = 100        # Maximum length of review
training_samples = 200
validation_samples = 10000
max_words = 10000   # Use only top 10000 words

tokenizer = Tokenizer(num_words = max_words) # Returns a list of lists
tokenizer.fit_on_texts(texts)    # Return One hot vectors
sequences = tokenizer.texts_to_sequences(texts)  
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


glove_dir = "E:\Aditya\AI_DL_ML\Datasets\GloVe"
embedding_index = {}
with open(os.path.join(glove_dir, 'glove.6B.100d.txt'), "r", encoding = "utf-8") as f:
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embedding_index[word] = coefs
        except UnicodeDecodeError:
            continue


print("Found %s word vectors" %(len(embedding_index)))

# We need to build an embedding matrix in order to be able to load it into an Embedding layer
# It must be a matrix of shape (max_words, embedding_dim), where each entry i contains
# the embedding_dim-dimensional vector for the word of index i in the reference word
# index (built during tokenization). Note that index 0 isn’t supposed to stand for any
# word or token—it’s a placeholder.

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length = maxlen))
model.add(Flatten())
model.add(Dense(33, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# The Embedding layer has a single weight matrix: a 2D float matrix where each entry i is
# the word vector meant to be associated with index i. Simple enough. Load the GloVe
# matrix you prepared into the Embedding layer, the first layer in the model.
# Additionally, you’ll freeze the Embedding layer (set its trainable attribute to False),

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False


# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,epochs=3,batch_size=32, verbose = -1, validation_data = (x_val, y_val))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


sequences = tokenizer.texts_to_sequences(test_texts)
x_tests = pad_sequences(sequences, maxlen = maxlen)
y_test = np.asarray(test_labels)
print(model.evaluate(x_tests, y_test, verbose = -1))
