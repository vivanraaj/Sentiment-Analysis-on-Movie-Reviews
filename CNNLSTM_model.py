# coding: utf-8

# importing required packages
import re
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import callbacks
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils

# fix random seed for reproducibility
np.random.seed(7)


####################################################
## Loading Data
####################################################

# initialize new lists
phrase = []
labels = []
test_phrase = []

# load training data
with open("train.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        phrase.append(row[2])
        labels.append(row[3])

# load testing data
with open("test.tsv") as testing:
    test = csv.reader(testing, delimiter="\t", quotechar='"')
    for s in test:
        test_phrase.append(s[2])


####################################################
## Data Preprocessing
####################################################

# function to preprocess the sentences of the reviews.
def clean_phrase(phrase):
    #Remove punctuation (with a regular expression) and convert to lower case
    words = (re.sub("[^a-zA-Z]", " ", phrase)).lower()
    return words

# remove the first row of the train dataset which is currently the header
del phrase[0]

# remove the first row of the testing dataset which is currently the header
del test_phrase[0]

# run preprocessing function on train dataset
clean_phrases = []
for x in phrase:
    new = clean_phrase(x)
    clean_phrases.append(new)
    
# run preprocessing function  on test dataset
test_clean_phrases = []
for xw in test_phrase:
    new_test = clean_phrase(xw)
    test_clean_phrases.append(new_test)

# join the rows as a string with '/n' as delimiter
all_text=' /n '.join(clean_phrases)
test_all_text=' /n '.join(test_clean_phrases)

# split each reviews of the training dataset and join them as a string
reviews = all_text.split(' /n ')
all_text = ' '.join(reviews)
# split each word of the training dataset in the string to a list
words = all_text.split()


# split each reviews of the training dataset and join them as a string
test_reviews = test_all_text.split(' /n ')
test_all_text = ' '.join(test_reviews)
# split each word of the training dataset in the string to a list
test_words = test_all_text.split()

# print no of rows for train and test 
print("Train reviews: {}".format(len(reviews)))
print("Test reviews: {}".format(len(test_reviews)))

# remove the first row of the labels which is currently the header
del labels[0]

# preprocessing on the label list
labels_cleaned = '\n'.join(labels)
labels_cleaned_last = labels_cleaned.split('\n')

# convert list to an array
labels_sentiment = [int(i) for i in labels_cleaned_last]
labels = np.array(labels_sentiment)

# combine the list that contains the individual words in the datasets
full_words = words + test_words

#create dictionaries that map the words in the vocabulary to integers. 
#Then we can convert each of our reviews into integers so they can be passed into the network.
from collections import Counter
counts = Counter(full_words)
vocab = sorted(counts, key=counts.get, reverse=True)

#Build a dictionary that maps words to integers
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


#Encode the words with integers. 
reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split( )])
    
test_reviews_ints = []
for eachs in test_reviews:
    test_reviews_ints.append([vocab_to_int[word] for word in eachs.split( )])


# check no of unique words in the corpus
# this will be the features to be extracted
print("No. of Features to be extracted: {}".format(len(vocab_to_int)))


review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


# check total no of rows not having zero length reviews
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

# remove zero length reviews
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])

#check again
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

#As maximum review length too many steps for RNN. Let's truncate to 12 steps. 
#For reviews shorter than 12 steps, we'll pad with 0s. For reviews longer than 12 steps,
# we will truncate them to the first 12 characters.
max_review_length = 12
X_train = sequence.pad_sequences(reviews_ints, maxlen=max_review_length)
x_test = sequence.pad_sequences(test_reviews_ints, maxlen=max_review_length)


# check shape of train input
print(X_train.shape)

# check shape of test input
print(x_test.shape)

# check no of unique words in the corpus
# Adding 1 because we use 0's for padding, dictionary started at 1
# this value will be passed to the embedding layer
top_words = len(vocab_to_int) + 1
print(top_words)

# One Hot Encoding the labels
y_train = np_utils.to_categorical(labels, 5)


####################################################
## Training
####################################################


# Creating Callbacks which is used in the Keras fit function
# ModelCheckpoints is used to save the model after every epoch
# EarlyStopping is used to stop training when the validation loss has not improved after 2 epochs
# Tensorboard is used tovisualize dynamic graphs of the training and test metrics
cbks = [callbacks.ModelCheckpoint(filepath='./checkpoint_model.h5', monitor='val_loss', save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=2),callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)]


#### Final Model Architecture built using Keras

# embedding layer size
embedding_vecor_length = 32

model = Sequential()
model.add(Embedding(19479, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# 1 layer of 100 units in the hidden layers of the LSTM cells
model.add(LSTM(100))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train,validation_split=0.20, epochs=5,verbose=1, batch_size=32,callbacks=cbks)


#to visualize the training graphs
#run "tensorboard --logdir='./logs' "  from the command terminal


####################################################
## Testing
####################################################

# load the saved model
# returns a compiled model
model = load_model('checkpoint_model.h5')


# visualize model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# run prediction
test_pred = model.predict_classes(x_test)


# edits the test file to input the prediction labels
test_df = pd.read_csv('test.tsv', sep='\t', header=0)
test_df['Sentiment'] = test_pred.reshape(-1,1) 
header = ['PhraseId', 'Sentiment']
test_df.to_csv('./predicted_values.csv', columns=header, index=False, header=True)