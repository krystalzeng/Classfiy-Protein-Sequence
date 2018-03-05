% load_ext autoreload
% autoreload 2
import numpy as np
from Model import *
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import *
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


def add_seq_len_feature(sequence, x):
    for i in range(len(sequences)):
        x[i] = len(sequences[i])

def all_amnio_acid(amino_acid, sequence, x):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = all(a in current_seq for a in amino_acid)
        feature[i] = 1.0 if matches else 0.0

    return np.column_stack((x,feature))

def amnio_acid_occurancy(amino_acid, sequence, x):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = sum(a in current_seq for a in amino_acid)
        feature[i] = matches

    return np.column_stack((x,feature))


def add_isoelectric_point(amino_acid, sequence, x):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        x[i] = np.ceil(analysis.isoelectric_point())
        # x[i] = feature
    # return np.column_stack((x,feature))

def amino_acids_percent(amino_acid, sequence, x):
    feature = np.zeros((len(sequences),20))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        index = 0
        precent_dict = analysis.get_amino_acids_percent()
        # arr = np.zeros(20)
        for aa in amino_acid:
            # arr[index] = precent_dict[aa] * 100
            # feature[i][index] = np.ceil(precent_dict[aa] * 100)
            feature[i][index] = precent_dict[aa] * 100
            index += 1
        # feature[i,:] = np.argsort(arr)

    return np.column_stack((x,feature))

def add_aromaticity(amino_acid, sequence, x):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = np.ceil(analysis.aromaticity() * 100)
    return np.column_stack((x,feature))

def add_instability_index(amino_acid, sequence, x):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.instability_index()
    return np.column_stack((x,feature))

def add_secondary_structure_fraction(amino_acid, sequence, x):
    feature = np.zeros((len(sequences),3))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        # feature[i,:] = np.asarray(analysis.secondary_structure_fraction())
        feature[i,:] = np.argsort(np.asarray(analysis.secondary_structure_fraction()))

    return np.column_stack((x,feature))


sequences, labels, Sequence_lookup_dict, Label_lookup_dict = read_data()
number_of_sequence = len(sequences)
data = np.zeros((number_of_sequence, 20))
y = np.zeros((number_of_sequence,1))
encode_label(labels, y)
add_seq_len_feature(sequences, data)

df = pd.DataFrame(data)
df
new_data = np.zeros((number_of_sequence, 20, 2))
for i in range(number_of_sequence):
    feature = np.zeros(20)
    current_seq = sequences[i].__str__()
    analysis = ProteinAnalysis(current_seq)
    precent_dict = analysis.get_amino_acids_percent()
    for index, element in enumerate(amino_acid):
        feature[index] = precent_dict[element] * 100
    new_data[i] = np.column_stack((data[i], feature))

data = new_data
add_isoelectric_point(amino_acid, sequences, data)
data[:10]
data.shape
train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.3, random_state=42)
train_x.shape
LSTM_model(train_x, train_y, test_x, test_y)
