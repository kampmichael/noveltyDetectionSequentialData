from distance_metrics import lcs as lcs_lib
from sklearn.neighbors import NearestNeighbors
import json
import numpy as np
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

def calcSensAndSpec(cm):
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    tp = cm[1,1]
    print(cm)
    TPR = float(tp)/float(tp + fn)
    TNR = float(tn)/float(tn + fp)
    return TPR, TNR

ACTIONS_FILE = "adfald_vocabulary.txt" # "yelpchi_vocab.txt" # "timeser_vocab.txt" # "fakenews_vocab.txt" # 
# has only id and sequence
TRAIN_FILE = "training_adfald.json" # "training_yelpchi.json" # "training_timeser.json" # "training_fakenews.json" # 
# includes field 'label' that is 1 for attack and 0 for normal sequence
TEST_FILE = "testing_adfald.json" # "testing_yelpchi.json" # "testing_timeser.json" # "testing_fakenews.json" # 
PAD_IDX = 341 # adfald # 6024 # yelpchi # 1001 # timeser2 # 5144 # fakenews # 
# longer sequences are completely stopping performance of LCS calculation 
max_len = 1000

def lcs(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
 
    # read a substring from the matrix
    result = []
    j = len(b)
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            result.append(a[i-1])
 
    if len(result) == 0:
        return 1.0
    else:
        return 1.0/len(result)

with open(TEST_FILE,"r") as f:
    test_data = json.load(f)
    print("Testing data size", len(test_data))

with open(TRAIN_FILE,"r") as f:
    train_data = json.load(f)
    print("Training data size", len(train_data))

# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i
actionIndex['PAD'] = PAD_IDX

train_sequences = []
#max_len = 0
for s in train_data:
    train_sequences.append(s['actionsQueue'])
    #if len(s['actionsQueue']) > max_len:
    #    max_len = len(s['actionsQueue'])
print("Train data size", len(train_sequences))
print("Maximum length", max_len)

test_sequences = []
test_labels = []
for s in test_data:
    test_sequences.append(s['actionsQueue'])
    test_labels.append(s['label'])
print("Test data size", len(test_sequences))

X_train = []
cut_s = 0
for s in train_sequences:
    tr_s = []
    for w in s:
        tr_s.append(actionIndex[w])
    if len(tr_s) < max_len:
        for j in range(max_len - len(tr_s)):
            tr_s.append(actionIndex['PAD'])
    elif len(tr_s) >= max_len:
        cut_s += 1
        tr_s = tr_s[:max_len]
    X_train.append(np.array(tr_s))
X_train = np.array(X_train)
print("Train data shape", X_train.shape)
print("Cut sequences", cut_s)

X_test = []
for s in test_sequences:
    tr_s = []
    for w in s:
        tr_s.append(actionIndex[w])
    if len(tr_s) < max_len:
        for j in range(max_len - len(tr_s)):
            tr_s.append(actionIndex['PAD'])
    elif len(tr_s) >= max_len:
        tr_s = tr_s[:max_len]
    X_test.append(np.array(tr_s))
X_test = np.array(X_test)
print("Test data shape", X_test.shape)

paramsAUCtotal = {}
for nn in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='auto', leaf_size=300) #, metric=lcs) # lcs_lib.bakkelund)
    nbrs.fit(X_train)
    distances, _ = nbrs.kneighbors(X_test)
    preds = distances.mean(axis=1)
    auc = roc_auc_score(test_labels, preds)
    print(nn, auc)
    paramsAUCtotal[auc] = [nn]

maxAuctotal = max(paramsAUCtotal.keys())
bestParamsAuctotal = paramsAUCtotal[maxAuctotal]
print("Best params: ", bestParamsAuctotal, " with AUC: ", maxAuctotal)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', leaf_size=300) #, metric=lcs)
nbrs.fit(X_train)
distances, _ = nbrs.kneighbors(X_test)
preds = distances.mean(axis=1)
np.save("knn_preds", preds)
np.save("knn_labels", test_labels)
auc = roc_auc_score(test_labels, preds)
_, _, thresholds = roc_curve(test_labels, preds)
thresh = thresholds[len(thresholds)//2]
cm = confusion_matrix(test_labels, [1 if p > thresh else 0 for p in preds])
print("Sensitivity and Specifity", calcSensAndSpec(cm))


# timeseries_2
# $0.94$ AUC, sensitivity $0.85$ and specificity is $0.60$