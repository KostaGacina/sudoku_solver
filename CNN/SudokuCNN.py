import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, MaxPooling2D, Reshape, Activation, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
import math
import random as rand

rand.seed(42)
data = pd.read_csv("data/sudoku_cnn.csv")

data = data[:250000]
puzzles = data["puzzle"]
labels = data["solution"]
puzzle_array = np.array([[c for c in puzzle] for puzzle in puzzles]).astype(int)
labels_array = np.array([[s for s in solution] for solution in labels]).astype(int)
puzzle_array = puzzle_array / 9 - 0.5
labels_array = labels_array - 1
print(puzzle_array)
labels_matrix = to_categorical(labels_array, num_classes=9)
print(data.shape)
train_indices = rand.sample((range(len(puzzle_array))), math.floor(0.8*len(data)))
valid_indices = np.setdiff1d(range(len(labels_array)), train_indices)

train_data = np.array([puzzle_array[i] for i in train_indices]).reshape(-1,9,9,1)
train_labels = labels_matrix[np.array(train_indices)].reshape(-1,9,9,9)

valid_data = np.array([puzzle_array[i] for i in valid_indices]).reshape(-1,9,9,1)
valid_labels = labels_matrix[np.array(valid_indices)].reshape(-1,9,9,9)

SudokuCNN = Sequential()
SudokuCNN.add(Input(shape=(9,9,1)))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same",))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 512, kernel_size=(3,3), strides=(1,1),use_bias=False, padding="same"))
SudokuCNN.add(BatchNormalization())
SudokuCNN.add(Activation("relu"))
SudokuCNN.add(Conv2D(filters = 9, kernel_size=(1,1), strides=(1,1),use_bias=True, padding="same", activation="softmax"))
SudokuCNN.compile(optimizer=Adam(learning_rate=1e-4),loss ="categorical_crossentropy", metrics=["categorical_accuracy"]) 