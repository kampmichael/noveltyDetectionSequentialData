import numpy as np
import lda
import json
from collections import OrderedDict
import joblib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

ACTIONS_FILE = "../yelpchi_vocab.txt" # "../fakenews_vocab.txt" # "timeser_vocab.txt" # "smsspam_vocab.txt" # "genom_vocab.txt" # "adfald_vocabulary.txt" # "old_amadeus_data_actions_list.txt"
CLUSTERS_FILE = "../categories.json"
TEST_FILE = "../testing_yelpchi.json" # "../testing_fakenews.json" # "testing_timeser.json" # "testing_smsspam.json" # "testing_genom.json" # "testing_adfald.json" # 

EMBEDDING_DIM = 256 # yelpchi # 256 # fakenews2 # 100 # glove_emb # 128 # timeser2 # 256 # timeser # 64 # smsspam # 128 # genom # 128 # adfald #
HIDDEN_DIM = 512 # yelpchi # 512 # fakenews2 # 512 # fakenews # 64 # timeser2 # 64 # timeser # 128 # smsspam # 128 # genom # # 512 # adfald #
BATCH = 64 # 8
PAD_IDX = 6024 # yelpchi # 5144 # fakenews # 1001 # timeser2 # 1081 # timeser # 914 # smsspam # 5 # genom # 341 # adfald # 
MAX_LEN = 2000
# max num of LDA for visualization
max_num = 9
CLEAN_STOPWORDS = False
CLEAN_VOCAB = "lda_fakenews_vocab.txt"

class LSTMNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = PAD_IDX)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sequences):
        #print(sequences.shape)
        embeds = self.word_embeddings(sequences)
        #print(embeds.shape)
        # lstm requires to have batch size on the second dimension
        lstm_out, _ = self.lstm(embeds.permute(1, 0, 2))
        #print(lstm_out.shape)
        # have to push out the batch size dimension on the first position for fully connected layer
        tag_space = self.hidden2tag(lstm_out.permute(1, 0, 2))
        #print(tag_space.shape)
        # need the classes (tag space, vocabulary size) on the second position for NLLLoss
        tag_scores = F.log_softmax(tag_space.permute(0, 2, 1), dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix, pad_len):
    if len(seq) > pad_len:
        seq = seq[:pad_len]
    idxs = [to_ix[w] for w in seq]
    for i in range(pad_len - len(idxs)):
        idxs.append(to_ix['PAD'])
    return torch.tensor(idxs, dtype=torch.long)

def perplexity(probs):
    product = 1
    count = 0
    for p in probs:
        # warning will be generated here when we overflow the value
        # best what we can do - assign the largest possible number to the product
        if math.isinf(product * (1.0/p)):
            product = 1e+300
        else:
            product *= (1.0/p)
        count += 1
    perpl = product**(1.0/count)
    return perpl

with open(TEST_FILE,"r") as f:
    test_data = json.load(f)
    print("Testing data size", len(test_data))

# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i

with open(CLUSTERS_FILE,"r") as f:
    clusters = json.load(f)

if CLEAN_STOPWORDS:
    cl_actionIndex = OrderedDict()
    file = open(CLEAN_VOCAB)
    for i, line in enumerate(file):
        cl_actionIndex[line.rstrip("\n")] = i
    # make bag of words from sequences
    wordDoc = np.zeros((len(test_data),len(cl_actionIndex),),dtype=np.int)
    for i in range(0,len(test_data)):
        actionsQueue = test_data[i]['actionsQueue']
        for j in range(0,len(actionsQueue)):
            action = actionsQueue[j]
            if cl_actionIndex.get(action) is None:
                continue
            wordIndex = cl_actionIndex[action]
            wordDoc[i][wordIndex] = wordDoc[i][wordIndex]+1
else:
    # make bag of words from sequences
    wordDoc = np.zeros((len(test_data),len(actionIndex),),dtype=np.int)
    for i in range(0,len(test_data)):
        actionsQueue = test_data[i]['actionsQueue']
        for j in range(0,len(actionsQueue)):
            action = actionsQueue[j]
            wordIndex = actionIndex[action]
            wordDoc[i][wordIndex] = wordDoc[i][wordIndex]+1

# adding PAD to vocabulary for lstms - not before translating to wordDoc that are used for LDA
# because LDA were trained without paddings
actionIndex['PAD'] = PAD_IDX

# load models
models = {}
for k in range(2,max_num):
    models["LDA" + str(k)] = joblib.load("LDA" + str(k) + "_model")

all_topic_names = []
for k in range(2,max_num):
    for t in range(k):
        all_topic_names.append("LDA" + str(k) + "Topic" + str(t))

cluster_topics = {}
for c in clusters:
    if c != 'Others':
        cluster_topics["cluster_" + c] = clusters[c]['groupInfo']
print(cluster_topics)

lstms = {}
for cl in cluster_topics:
    model = LSTMNet(EMBEDDING_DIM, HIDDEN_DIM, len(actionIndex), len(actionIndex)).cuda()
    model.load_state_dict(torch.load(cl + "lstm_model"))
    model.eval()
    lstms[cl] = model
global_lstm = LSTMNet(EMBEDDING_DIM, HIDDEN_DIM, len(actionIndex), len(actionIndex)).cuda()
global_lstm.load_state_dict(torch.load("global_lstm_model"))
global_lstm.eval()

test_labels = []
cluster_perplexities = []
global_perplexities = []
per_cluster_perplexities = {}
for j in range(len(wordDoc)):
    sequence = wordDoc[j]
    predictions = {}
    for k in range(2,max_num):
        probs = models["LDA" + str(k)].transform(sequence)[0]
        for i, p in enumerate(probs):
            predictions["LDA" + str(k) + "Topic" + str(i)] = p

    max_score = 0.0
    for ct in cluster_topics:
        score = 0.0
        for c in cluster_topics[ct]:
            score += predictions[c]
        score /= len(cluster_topics[ct])
        if score > max_score:
            max_score = score
            belong_cluster = ct

    s = test_data[j]['actionsQueue']
    test_labels.append(test_data[j]['label'])
    if len(s) > MAX_LEN:
        s_len = MAX_LEN
    else:
        s_len = len(s)
    s_tensor = prepare_sequence(s, actionIndex, s_len)
    tag_scores = lstms[belong_cluster](s_tensor.cuda().view(1,-1))
    tag_scores = tag_scores.squeeze().exp().data.cpu().numpy()
    cl_probs = []
    for i,t in enumerate(s_tensor.data.numpy()[1:]):
        #print("-------------", t)
        #print("max probability", tag_scores[:,i].max())
        #print("prediction", tag_scores[:,i].argmax())
        #print("probability of target", tag_scores[:,i][t])
        cl_probs.append(tag_scores[:,i][t])
    #print("Cluster: Perplexity", perplexity(cl_probs), "for label", test_data[j]['label'])
    cluster_perplexities.append(perplexity(cl_probs))
    tag_scores = global_lstm(s_tensor.cuda().view(1,-1))
    tag_scores = tag_scores.squeeze().exp().data.cpu().numpy()
    gl_probs = []
    for i,t in enumerate(s_tensor.data.numpy()[1:]):
        #print("-------------", t)
        #print("max probability", tag_scores[:,i].max())
        #print("prediction", tag_scores[:,i].argmax())
        #print("probability of target", tag_scores[:,i][t])
        gl_probs.append(tag_scores[:,i][t])
    #print("Global: Perplexity", perplexity(gl_probs), "for label", test_data[j]['label'])
    global_perplexities.append(perplexity(gl_probs))
    if per_cluster_perplexities.get(belong_cluster) is None:
        per_cluster_perplexities[belong_cluster] = [[test_labels[j], 
                                        perplexity(cl_probs), perplexity(gl_probs)]]
    else:
        per_cluster_perplexities[belong_cluster].append([test_labels[j], 
                                        perplexity(cl_probs), perplexity(gl_probs)])

# save per cluster predictions
# dictionary with cluster names as keys, each contains sequences related to the cluster
# each sequence has [label, cluster_score, global_score]
with open("per_cluster_perplexities.json","w") as outp:
    json.dump(per_cluster_perplexities, outp)

for i in range(len(test_labels)):
    print("Label", test_labels[i], "Cluster", cluster_perplexities[i],
            "Global", global_perplexities[i])

np.save("test_labels", test_labels)
np.save("cluster_perplexities", cluster_perplexities)
np.save("global_perplexities", global_perplexities)

