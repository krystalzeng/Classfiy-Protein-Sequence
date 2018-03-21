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
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

def Logistic_regression(X, y, train = False):
    lr = linear_model.LogisticRegression()
    if train:
        lr.fit(X,y.ravel())
    return lr

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

    return score

def test_MLP(X, y):
    MLP_settings = [(1,32), (2,32), (3,32), (1,64), (2,64), (3,64), (1,128), (2,128), (3,128)]
    MLP_accs = []

    for setting in MLP_settings:
        layers = setting[0]
        units = setting[1]
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.125)
        score = MLP(train_x, train_y, test_x, test_y, units, layers)
        result = (layers + 2, units, score)
        MLP_accs.append(result)
        print('Score', score)
    return MLP_accs

def kfold_cross_validation(model, X, y, model_name, splits = 8):
    Test_accs = []
    f1_scores = []
    kf = KFold(n_splits=splits, shuffle=True)
    cnf_matrices = np.zeros((4,4))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        current_model = model.fit(X_train, y_train.ravel())
        predicted_y = current_model.predict(X_test)
        test_acc = current_model.score(X_test, y_test.ravel())
        Test_accs.append(test_acc)
        cnf_matrices += confusion_matrix(y_test.ravel(), predicted_y)
        f1_scores.append(f1_score(y_test.ravel(), predicted_y, average='weighted'))

    np.set_printoptions(precision=2)
    class_names = ['cyto', 'mito', 'nucleus', 'secreted']

    # Plot normalized confusion matrix
    normalized_fig = plt.figure(figsize=(8, 8))
    plot_confusion_matrix(cnf_matrices, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    normalized_fig.savefig(model_name+'_normalized.png', bbox_inches="tight")
    return Test_accs, f1_scores


def linear_svm():
    return LinearSVC(random_state=0)

def random_forest(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)
    clf.fit(train_x, train_y)
    predictions = clf.predict(test_x)
    correct_prediction = np.equal(predictions, test_y)
    test_acc = np.mean(correct_prediction)
    return test_acc

def random_baseline_model(X, y, splits = 8):
    Test_accs = []
    f1_scores = []
    kf = KFold(n_splits=splits, shuffle=True)
    cnf_matrices = np.zeros((4,4))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predicted_y = np.random.randint(4, size=len(y_test))
        correct_prediction = np.equal(predicted_y, y_test)
        test_acc = np.mean(correct_prediction)
        Test_accs.append(test_acc)
        cnf_matrices += confusion_matrix(y_test.ravel(), predicted_y)
        f1_scores.append(f1_score(y_test.ravel(), predicted_y, average='weighted'))

    np.set_printoptions(precision=2)
    class_names = ['cyto', 'mito', 'nucleus', 'secreted']

    # Plot normalized confusion matrix
    normalized_fig = plt.figure(figsize=(8, 8))
    plot_confusion_matrix(cnf_matrices, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    normalized_fig.savefig('uniform_normalized.png', bbox_inches="tight")

    return Test_accs, f1_scores

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
