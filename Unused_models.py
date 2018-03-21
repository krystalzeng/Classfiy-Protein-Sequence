
#
# x_axis = []
# y_axis = []
# z_axis = []
#
# for acc in MLP_accs:
#     x_axis.append(acc[0])
#     y_axis.append(acc[1])
#     z_axis.append(acc[2][1])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# surf = ax.plot_trisurf(np.asarray(x_axis), np.asarray(y_axis), np.asarray(z_axis), cmap=cm.jet, linewidth=0)
# fig.colorbar(surf)
#
# ax.xaxis.set_major_locator(MaxNLocator(5))
# ax.yaxis.set_major_locator(MaxNLocator(6))
# ax.zaxis.set_major_locator(MaxNLocator(5))
#
# fig.tight_layout()
#
# plt.show()
# fig.savefig('MLP_analysis.png')
# z_axis
# y_axis
# tb = pd.DataFrame()
# tb['Number of layers'] = pd.Series(x_axis)
# tb['Number of units'] = pd.Series(y_axis)
# tb['Test accuracy'] = pd.Series(z_axis)
#
# ax = plt.subplot(figsize=(12, 2), frame_on=False) # no visible frame
# ax.xaxis.set_visible(False)  # hide the x axis
# ax.yaxis.set_visible(False)  # hide the y axis
#
# table(ax, tb)  # where df is your data frame
#
# plt.savefig('mytable.png')
# lsvm = linear_svm(train_x, train_y.ravel(), test_x, test_y.ravel())
# correct_prediction = np.equal(np.argmax(lsvm, 1), test_y)
# test_acc = np.mean(correct_prediction)
# print('SVM acc', test_acc)

def knear(train_x, train_y, test_x, test_y):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(train_x)
    distances, indices = nbrs.kneighbors(test_x)
    return distances, indices

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
add_seq_len_feature(sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['Sequence length'] = logit_score

# An uniform model is implemented to act as the baseline for this experiment

# all_amnio_acid(amino_acid, sequences, X_df)
# X = X_df.values
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
# logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
# print('Logistic score', logit_score)
# score_hisotry['If contain all aa'] = logit_score


amnio_acid_occurancy(amino_acid, sequences, X_df)
X = X_df.values
X_df
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['aa occurancy'] = logit_score

add_isoelectric_point(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['Isoelectric Point'] = logit_score


amino_acids_percent(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['amino_acids_percent'] = logit_score


add_aromaticity(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['Aromaticity'] = logit_score


add_secondary_structure_fraction(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['secondary_structure_fraction'] = logit_score


add_molecular_weight(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['molecular_weight'] = logit_score


add_LCC(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['LCC'] = logit_score

local_amnio_acid(amino_acid, sequences, X_df)
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)
logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)
score_hisotry['Local aa'] = logit_score
X_df.shape
X_df

add_local_test(amino_acid, sequences, X_df)

X = X_df.values
cl = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
lr_accs = kfold_cross_validation(cl, X, y)
accs
avg = np.sum(accs) / 8
avg
svm = linear_svm(train_x, train_y.ravel(), test_x, test_y.ravel())
svm_accs = kfold_cross_validation(svm, X, y)
svm_accs

unifrom_accs = random_baseline_model(X, y)
unifrom_accs

MLP_accs = test_MLP(X,y)
MLP_accs


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
rf = random_forest(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Random forest acc', rf)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)
rf = random_forest(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Random forest acc', rf)
