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
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM
from keras import regularizers

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def Logistic_regression(train_x, train_y, test_x, test_y):
    LogisticReg = linear_model.LogisticRegression()
    LogisticReg.fit(train_x, train_y)
    test_acc = LogisticReg.score(test_x, test_y)
    predicted_y = LogisticReg.predict(test_x)


    cnf_matrix = confusion_matrix(test_y, predicted_y)
    np.set_printoptions(precision=2)
    class_names = ['cyto', 'mito', 'nucleus', 'secreted']

    # Plot non-normalized confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')


    # Plot normalized confusion matrix
    normalized_fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    fig.savefig('Logistic_Regression_unnormalized.png')
    normalized_fig.savefig('Logistic_Regression_normalized.png')

    return test_acc

def MLP(train_x, train_y, test_x, test_y, hidden_units = 64, number_of_layers = 1):
    train_y_one_hot = keras.utils.to_categorical(train_y, num_classes=4)
    test_y_one_hot = keras.utils.to_categorical(test_y, num_classes=4)

    feature_size = train_x.shape[1]
    output_size = train_y_one_hot.shape[1]
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_dim=feature_size))
    model.add(Dropout(0.5))
    for layer in range(number_of_layers):
        model.add(Dense(hidden_units, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 128
    model.fit(train_x, train_y_one_hot,
              epochs=20,
              batch_size=batch_size)
    score = model.evaluate(test_x, test_y_one_hot, batch_size=batch_size)

    # predicted_y = model.predict(test_x)
    # cnf_matrix = confusion_matrix(test_y, predicted_y)

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

def random_forest(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    correct_prediction = np.equal(predictions, test_y)
    test_acc = np.mean(correct_prediction)
    return test_acc

def random_baseline_model(train_x, train_y, test_x, test_y):

    predicted_y = np.zeros(len(test_y))
    index = 0
    for x in test_x:
        predicted_y[index] = np.random.randint(4)

    # correct_prediction = np.equal(predicted_y, test_y)
    # test_acc = np.mean(correct_prediction)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_y, predicted_y)
    np.set_printoptions(precision=2)
    class_names = ['0', '1', '2', '3']

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
