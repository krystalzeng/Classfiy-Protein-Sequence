% load_ext autoreload
% autoreload 2
import numpy as np
from Processing import *
from Model import *
from collections import defaultdict
from Bio import SeqIO

amino_acid = ['R', 'K', 'D', 'E', 'Q', 'N', 'H', 'S', 'T','Y', 'C', 'W', 'A', 'I', 'L', 'M', 'F', 'V', 'P', 'G']

score_hisotry = defaultdict()
sequences, labels, Sequence_lookup_dict, Label_lookup_dict, test_sequences, test_sequence_ids = read_data()
number_of_sequence = len(sequences)
y = np.zeros((number_of_sequence,1))
encode_label(labels, y)


train_X_df = pd.DataFrame()
add_all_features(amino_acid, sequences, train_X_df)
train_X = train_X_df.values
trained_lr = Logistic_regression(train_X, y)
lr_accs, lr_f1 = kfold_cross_validation(trained_lr, train_X, y, 'lr')
trained_lr = Logistic_regression(train_X, y, True)

test_X_df = pd.DataFrame()

add_all_features(amino_acid, test_sequences, test_X_df)

test_X_df
predicted_p = trained_lr.predict_proba(test_X_df.values)


test_labels = defaultdict()
test_labels[0] = 'cyto'
test_labels[1] = 'mito'
test_labels[2] = 'nucleus'
test_labels[3] = 'secreted'

try:
    f = open('result.txt','w+')
    for i, id in enumerate(test_sequence_ids):
        f.write(id+'  ')
        f.write(test_labels[int(np.argmax(predicted_p[i]))]+'  ')
        f.write(str(int(np.amax(predicted_p[i])*100)))
        f.write('%\n')
    f.close()

except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except ValueError:
    print("Could not convert data to an integer.")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
