import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model, svm
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.cross_validation import train_test_split
np.set_printoptions(threshold=np.inf)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
# from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM

def Logistic_regression(train_x, train_y, test_x, test_y):
    LogisticReg = linear_model.LogisticRegression()
    LogisticReg.fit(train_x, train_y)
    test_acc = LogisticReg.score(test_x, test_y)
    return test_acc

def MLP(train_x, train_y, test_x, test_y):
    train_y_one_hot = keras.utils.to_categorical(train_y, num_classes=4)
    test_y_one_hot = keras.utils.to_categorical(test_y, num_classes=4)

    feature_size = train_x.shape[1]
    output_size = train_y_one_hot.shape[1]
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=feature_size))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 128
    model.fit(train_x, train_y_one_hot,
              epochs=50,
              batch_size=batch_size)
    score = model.evaluate(test_x, test_y_one_hot, batch_size=batch_size)
    return score

def Conv(train_x, train_y, test_x, test_y):
    feature_size = train_x.shape[2]
    output_size = train_y.shape[1]
    seq_length = train_x.shape[1]
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, feature_size)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=128, epochs=10)
    score = model.evaluate(test_x, test_y, batch_size=128)

    return score

def LSTM_model(train_x, train_y, test_x, test_y):
    feature_size = train_x.shape[2]
    output_size = 4
    seq_length = train_x.shape[1]
    train_y_one_hot = keras.utils.to_categorical(train_y, num_classes=4)
    test_y_one_hot = keras.utils.to_categorical(test_y, num_classes=4)

    model = Sequential()
    model.add(LSTM(64,input_shape=(seq_length,feature_size),
               return_sequences=True))
    # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(64))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_x, train_y_one_hot,
              batch_size=128, epochs=10,
              validation_data=(test_x, test_y_one_hot))

    lstm_score = model.evaluate(test_x, test_y_one_hot, batch_size=128)

    return lstm_score

def linear_svm(train_x, train_y, test_x, test_y):
    lin_clf = svm.SVC(decision_function_shape='ovo')
    lin_clf.fit(train_x, train_y)
    dec = lin_clf.decision_function(test_x)
    return dec

def knear(train_x, train_y, test_x, test_y):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(train_x)
    distances, indices = nbrs.kneighbors(test_x)

    return distances, indices
