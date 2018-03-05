% load_ext autoreload
% autoreload 2
import numpy as np
from Model import *
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import *
from Bio.SeqUtils.lcc import *
from collections import defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
np.set_printoptions(threshold=np.inf)
import pandas as pd

amino_acid = ['R', 'K', 'D', 'E', 'Q', 'N', 'H', 'S', 'T','Y', 'C', 'W', 'A', 'I', 'L', 'M', 'F', 'V', 'P', 'G']

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def blind_test():
    file_object  = open('Blind_test.txt', 'r')
    blind_test_file = file_object.read()

def drop_column(col_name, X_df):
    X_df.drop([col_name], axis=1)

def read_data():
    Label_lookup_dict = defaultdict()
    Sequence_lookup_dict = defaultdict()
    sequences = []
    labels = []

    for seq_record in SeqIO.parse("cyto.fasta", "fasta"):
        sequences.append(seq_record.seq)
        labels.append('cyto')
        Sequence_lookup_dict[seq_record.id] = seq_record.seq
        Label_lookup_dict[seq_record.id] = 'cyto'

    for seq_record in SeqIO.parse("mito.fasta", "fasta"):
        sequences.append(seq_record.seq)
        labels.append('mito')
        Sequence_lookup_dict[seq_record.id] = seq_record.seq
        Label_lookup_dict[seq_record.id] = 'mito'

    for seq_record in SeqIO.parse("nucleus.fasta", "fasta"):
        sequences.append(seq_record.seq)
        labels.append('nucleus')
        Sequence_lookup_dict[seq_record.id] = seq_record.seq
        Label_lookup_dict[seq_record.id] = 'nucleus'

    for seq_record in SeqIO.parse("secreted.fasta", "fasta"):
        sequences.append(seq_record.seq)
        labels.append('secreted')
        Sequence_lookup_dict[seq_record.id] = seq_record.seq
        Label_lookup_dict[seq_record.id] = 'secreted'

    return sequences, labels, Sequence_lookup_dict, Label_lookup_dict

def encode_label(labels, y):
    for i in range(len(labels)):
        if labels[i] == 'cyto':
            y[i] = 0
        if labels[i] == 'mito':
            y[i] = 1
        if labels[i] == 'nucleus':
            y[i] = 2
        if labels[i] == 'secreted':
            y[i] = 3


def add_seq_len_feature(sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        feature[i] = len(sequences[i])
    X_df['Sequence length'] = pd.Series(feature)

def all_amnio_acid(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = all(a in current_seq for a in amino_acid)
        feature[i] = 1.0 if matches else 0.0

    X_df['If contain all aa'] = pd.Series(feature)

def amnio_acid_occurancy(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = sum(a in current_seq for a in amino_acid)
        feature[i] = matches

    X_df['aa occurancy'] = pd.Series(feature)


def add_isoelectric_point(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.isoelectric_point()

    X_df['Isoelectric Point'] = pd.Series(feature)

def amino_acids_percent(amino_acid, sequence, X_df):
    feature = np.zeros((len(sequences),20))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        index = 0
        precent_dict = analysis.get_amino_acids_percent()
        for aa in amino_acid:
            feature[i][index] = precent_dict[aa] * 100
            index += 1

    i = 0
    for aa in amino_acid:
        X_df[aa] = pd.Series(feature[:,i])
        i +=1

def add_aromaticity(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.aromaticity() * 100

    X_df['Aromaticity'] = pd.Series(feature)

def add_instability_index(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.instability_index()

    X_df['Instability index'] = pd.Series(feature)

def add_secondary_structure_fraction(amino_acid, sequence, X_df):
    feature = np.zeros((len(sequences),3))
    names = ['Helix', 'Turn', 'Sheet']
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        # feature[i,:] = np.asarray(analysis.secondary_structure_fraction())
        feature[i,:] = np.argsort(np.asarray(analysis.secondary_structure_fraction()))
    i = 0
    for m in names:
        X_df[m] = pd.Series(feature[:,i])
        i +=1

def add_molecular_weight(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        try:
            if molecular_weight(current_seq, seq_type='protein'):
                # feature[i] = molecular_weight(current_seq, seq_type='protein')
                feature[i] = 1.0
            else:
                feature[i] = 0.0
        except ValueError:
            feature[i] = 0.0

    X_df['Molecular weight'] = pd.Series(feature)

def add_LCC(amino_acid, sequence, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        feature[i] = lcc_simp(current_seq)

    X_df['LCC'] = pd.Series(feature)


sequences, labels, Sequence_lookup_dict, Label_lookup_dict = read_data()
number_of_sequence = len(sequences)
y = np.zeros((number_of_sequence,1))
encode_label(labels, y)
X_df = pd.DataFrame()

add_seq_len_feature(sequences, X_df)
all_amnio_acid(amino_acid, sequences, X_df)
amnio_acid_occurancy(amino_acid, sequences, X_df)
add_isoelectric_point(amino_acid, sequences, X_df)
amino_acids_percent(amino_acid, sequences, X_df)
add_aromaticity(amino_acid, sequences, X_df)
add_secondary_structure_fraction(amino_acid, sequences, X_df)
add_molecular_weight(amino_acid, sequences, X_df)
add_LCC(amino_acid, sequences, X_df)
X_df
X = X_df.values
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)


logit_score = Logistic_regression(train_x, train_y.ravel(), test_x, test_y.ravel())
print('Logistic score', logit_score)

score = MLP(train_x, train_y, test_x, test_y)
print('Score', score)

lsvm = linear_svm(train_x, train_y.ravel(), test_x, test_y.ravel())
correct_prediction = np.equal(np.argmax(lsvm, 1), test_y)
test_acc = np.mean(correct_prediction)
print('SVM acc', test_acc)
