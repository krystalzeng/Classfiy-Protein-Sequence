# % load_ext autoreload
# % autoreload 2
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import *
from Bio.SeqUtils.lcc import *
from collections import defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
np.set_printoptions(threshold=np.inf)
import pandas as pd
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pandas.tools.plotting import table
from sklearn.metrics import confusion_matrix

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def drop_column(col_name, X_df):
    return X_df.drop([col_name], axis=1)

def read_data():
    Label_lookup_dict = defaultdict()
    Sequence_lookup_dict = defaultdict()
    sequences = []
    labels = []
    test_squences = []
    test_squences_ids = []

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

    for seq_record in SeqIO.parse("Blind_test.fasta", "fasta"):
        test_squences.append(seq_record.seq)
        test_squences_ids.append(seq_record.id)

    return sequences, labels, Sequence_lookup_dict, Label_lookup_dict, test_squences, test_squences_ids

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


def add_seq_len_feature(sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        feature[i] = len(sequences[i])
    X_df['Sequence length'] = pd.Series(feature)

def all_amnio_acid(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = all(a in current_seq for a in amino_acid)
        feature[i] = 1.0 if matches else 0.0

    X_df['If contain all aa'] = pd.Series(feature)

def local_amnio_acid(amino_acid, sequences, X_df):
    feature_front = np.zeros((len(sequences),20))
    feature_end = np.zeros((len(sequences),20))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        index = 0
        for aa in amino_acid:
            feature_front[i,index] = current_seq[:50].count(aa)
            feature_end[i,index] = current_seq[-50:].count(aa)
            index += 1

    i = 0
    for aa in amino_acid:
        X_df['Local first 50 '+ aa] = pd.Series(feature_front[:,i])
        X_df['Local last 50 '+ aa] = pd.Series(feature_end[:,i])
        i +=1

def amnio_acid_occurancy(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = sum(a in current_seq for a in amino_acid)
        feature[i] = matches

    X_df['aa occurancy'] = pd.Series(feature)


def add_isoelectric_point(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.isoelectric_point()

    X_df['Isoelectric Point'] = pd.Series(feature)

def amino_acids_percent(amino_acid, sequences, X_df):
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

def add_aromaticity(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.aromaticity() * 100

    X_df['Aromaticity'] = pd.Series(feature)

def add_instability_index(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        analysis = ProteinAnalysis(current_seq)
        feature[i] = analysis.instability_index()

    X_df['Instability index'] = pd.Series(feature)

def add_secondary_structure_fraction(amino_acid, sequences, X_df):
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

def add_molecular_weight(amino_acid, sequences, X_df):
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

def add_LCC(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        feature[i] = lcc_simp(current_seq)

    X_df['LCC'] = pd.Series(feature)

def add_local_beginning_pattern(amino_acid, sequences, X_df):
    positions = 5
    feature = np.zeros((len(sequences),positions))

    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        index = 0
        for letter in current_seq[:positions]:
            if letter != 'X' and letter != 'U':
                feature[i][index] = amino_acid.index(letter)
            else:
                feature[i][index] = -1
            index += 1

    for j in range(positions):
        X_df['Position'+str(j)] = pd.Series(feature[:,j])

def add_local_ending_pattern(amino_acid, sequences, X_df):
    positions = 5
    feature = np.zeros((len(sequences),positions))

    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        index = 0
        for letter in current_seq[-positions:]:
            if letter != 'X' and letter != 'U':
                feature[i][index] = amino_acid.index(letter)
            else:
                feature[i][index] = -1
            index += 1

    for j in range(positions):
        X_df['Last Position'+str(j)] = pd.Series(feature[:,j])

def add_number_of_ambiguous_aa(amino_acid, sequences, X_df):
    feature = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        matches = sum(a not in amino_acid for a in current_seq)
        feature[i] = matches

    X_df['Ambiguous_aa'] = pd.Series(feature)

def add_ambiguous_aa(amino_acid, sequences, X_df):
    feature_x = np.zeros(len(sequences))
    feature_u = np.zeros(len(sequences))
    for i in range(len(sequences)):
        current_seq = sequences[i].__str__()
        feature_x[i] = sum(a == 'X' for a in current_seq)
        feature_u[i] = sum(a == 'U' for a in current_seq)

    X_df['Ambiguous_aa_U'] = pd.Series(feature_u)
    X_df['Ambiguous_aa_X'] = pd.Series(feature_x)


def add_all_features(amino_acid, sequences, X_df):

    add_seq_len_feature(sequences, X_df)

    amnio_acid_occurancy(amino_acid, sequences, X_df)

    add_isoelectric_point(amino_acid, sequences, X_df)

    amino_acids_percent(amino_acid, sequences, X_df)

    add_aromaticity(amino_acid, sequences, X_df)

    add_secondary_structure_fraction(amino_acid, sequences, X_df)

    add_molecular_weight(amino_acid, sequences, X_df)

    add_LCC(amino_acid, sequences, X_df)

    local_amnio_acid(amino_acid, sequences, X_df)

    add_local_beginning_pattern(amino_acid, sequences, X_df)

    add_local_ending_pattern(amino_acid, sequences, X_df)

    # return X_df, X_df.values


# X_df.shape
# X = X_df.values
# cl = Logistic_regression()
# lr_accs, lr_f1 = kfold_cross_validation(cl, X, y, 'lr')
# lr_accs
# lr_f1
# np.sum(lr_accs) / 8
# np.sum(lr_f1) / 8
#
#
# svm = linear_svm()
# svm_accs, svm_f1 = kfold_cross_validation(svm, X, y, 'svm')
# svm_accs
# svm_f1
# np.sum(svm_accs) / 8
# np.sum(svm_f1) / 8
#
# unifrom_accs, uniform_f1 = random_baseline_model(X, y)
# unifrom_accs
# np.sum(uniform_f1)/8
#
# MLP_accs = test_MLP(X,y)
# MLP_accs
#
# sequences = "MESKGASSCRLLFCLLISATVFRPGLGWYTVNSAYGDTIIIPCRLDVPQNLMFGKWKYEK\
# PDGSPVFIAFRSSTKKSVQYDDVPEYKDRLNLSENYTLSISNARISDEKRFVCMLVTEDN\
# VFEAPTIVKVFKQPSKPEIVSKALFLETEQLKKLGDCISEDSYPDGNITWYRNGKVLHPL\
# EGAVVIIFKKEMDPVTQLYTMTSTLEYKTTKADIQMPFTCSVTYYGPSGQKTIHSEQAVF\
# DIYYPTEQVTIQVLPPKNAIKEGDNITLKCLGNGNPPPEEFLFYLPGQPEGIRSSNTYTL\
# TDVRRNATGDYKCSLIDKKSMIASTAITVHYLDLSLNPSGEVTRQIGDALPVSCTISASR\
# NATVVWMKDNIRLRSSPSFSSLHYQDAGNYVCETALQEVEGLKKRESLTILVEGKPQIKM\
# TKKTDPSGLSKTIICHVEGFPKPAIQWTITGSGSVINQTEESPYINGRYYSKIIISPEEN\
# VTLTCTAENQLERTVNSLNVSAISIPEHDEADEISDENREKVNDQAKLIVGIVVGLLLAA\
# LVAGVVYWLYMKKSKTASKHVNKDLGNMEENKKLEENNHKTEA"
#
# cl = Logistic_regression()
# cl.predict(sequences)
# f = open("Blind_test.txt")
# blind_test_raw = f.readlines()
# blind_test = defaultdict(str)
# for lines in blind_test_raw:
#     if lines[0] = '>':
#         blind_test[lines]
