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

samples_dataframe = samples_dataframe.reindex(np.random.permutation(samples_dataframe.index))
print(samples_dataframe)

samples = []
labels = []

with open("NNAVData", 'r') as f:
	reader = csv.reader(f)
	samples = list(reader)

shuffle(samples)

for x in range(len(samples)):
	itemlabel = samples[x][257]
	labels.append(itemlabel)
	del samples[x][257]
	for y in range(len(samples[x])):
		samples[x][y] = int(samples[x][y])

for z in range(len(labels)):
	if labels[z] == "SAFE":
		labels[z] = int(0)
	else:
		labels[z] = int(1)

print("Number of samples: " + str(len(samples)))
print("Number of labels: " + str(len(labels)))

train_data = samples[:4000]
train_data = np.asarray(train_data)
train_data = keras.preprocessing.sequence.pad_sequences(train_data)
train_labels = labels[:4000]

test_data = samples[4000:]
test_data = np.asarray(test_data)
test_data = keras.preprocessing.sequence.pad_sequences(test_data)
test_labels = labels[4000:]

factors = 256

model = keras.Sequential()
#Reducing embedding size reduces RAM requirements
model.add(keras.layers.Embedding(20000000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
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

model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=512, validation_data=(x_val, y_val), verbose=2)

results = model.evaluate(test_data, test_labels)

print("printing results")
print(model.metrics_names)
print(results)






