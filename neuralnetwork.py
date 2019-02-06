#!/usr/bin/python3


from __future__ import print_function

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math
import csv
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.data import Dataset
from random import shuffle

samples_dataframe = pd.read_csv("NNAVData", sep=',')
#print(samples_dataframe)
#print(type(samples_dataframe))

samples_dataframe = samples_dataframe.reindex(np.random.permutation(samples_dataframe.index))
print(samples_dataframe)

samples = []
labels = []

with open("NNAVData", 'r') as f:
	reader = csv.reader(f)
	samples = list(reader)

shuffle(samples)
#print(samples)

for x in range(len(samples)):
	itemlabel = samples[x][257]
	labels.append(itemlabel)
	del samples[x][257]
	for y in range(len(samples[x])):
		samples[x][y] = int(samples[x][y])
	#templist = list()
	#templist.append(samples[x])
	#samples[x] = templist

for z in range(len(labels)):
	if labels[z] == "SAFE":
		#print("TRIGGER0")
		labels[z] = int(0)
	else:
		#print("TRIGGER1")
		labels[z] = int(1)

#print(samples)


#print(samples)
#print(labels)

print("Number of samples: " + str(len(samples)))
print("Number of labels: " + str(len(labels)))
#print(type(samples[2][2]))

#from keras.preprocessing.text import Tokenizer
#tk = Tokenizer()

train_data = samples[:4000]
train_data = np.asarray(train_data)
train_data = keras.preprocessing.sequence.pad_sequences(train_data)
train_labels = labels[:4000]

test_data = samples[4000:]
test_data = np.asarray(test_data)
test_data = keras.preprocessing.sequence.pad_sequences(test_data)
test_labels = labels[4000:]

#print("Type of train data index index")
#print(type(train_data[2][2]))
#print("Type of train data index")
#print(type(train_labels[2]))
#print("Type of test data index index")
#print(type(test_data[2][2]))
#print("Type of test data index")
#print(type(test_labels[2]))


#print(train_data[0])
#print(train_labels[0])
#print(test_data[0])
#print(test_labels[0])

factors = 256

model = keras.Sequential()
#Reducing embedding size reduces RAM requirements
model.add(keras.layers.Embedding(20000000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(), loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:256]
partial_x_train = train_data[256:]

y_val = train_labels[:256]
partial_y_train = train_labels[256:]

#history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=256, validation_data=(x_val, y_val), verbose=2)

#print(test_data)

results = model.evaluate(test_data, test_labels)

print("printing results")
print(model.metrics_names)
print(results)









