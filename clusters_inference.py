import numpy as np
import lda
import json
from collections import OrderedDict
import joblib

DATA_FILE = "used_data_lda.json"
ACTIONS_FILE = "lda_fakenews_vocab.txt" # "lda_yelpchi_vocab.txt" # "timeser_vocab.txt" # "smsspam_vocab.txt" # "yelpchi_vocab.txt" # "adfald_vocabulary.txt" # "old_amadeus_data_actions_list.txt"
CLUSTERS_FILE = "categories.json"
# max num of LDA for visualization
max_num = 9

with open(DATA_FILE,"r") as f:
    data = json.load(f)
    print("Data size", len(data))
    
# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i

with open(CLUSTERS_FILE,"r") as f:
    clusters = json.load(f)

# make bag of words from sequences
wordDoc = np.zeros((len(data),len(actionIndex),),dtype=np.int)
for i in range(0,len(data)):
    actionsQueue = data[i]['actionsQueue']
    for j in range(0,len(actionsQueue)):
        action = actionsQueue[j]
        if actionIndex.get(action) is None:
            continue
        wordIndex = actionIndex[action]
        wordDoc[i][wordIndex] = wordDoc[i][wordIndex]+1

# load models
models = {}
for k in range(2,max_num):
    models["LDA" + str(k)] = joblib.load("LDA" + str(k) + "_model")

all_topic_names = []
for k in range(2,max_num):
    for t in range(k):
        all_topic_names.append("LDA" + str(k) + "Topic" + str(t))

cluster_topics = {}
all_except_others = []
for c in clusters:
    cluster_topics["cluster_" + c] = clusters[c]['groupInfo']
    all_except_others += clusters[c]['groupInfo']
cluster_topics["cluster_Others"] = list(set(all_topic_names) - set(all_except_others))
print(cluster_topics)

# the cluster separation from the interface, to check the correctness of downloaded data
correct = 0
for j in range(len(wordDoc)):
    sequence = wordDoc[j]
    predictions = {}
    for k in range(2,max_num):
        probs = models["LDA" + str(k)].transform(sequence)[0]
        for i, p in enumerate(probs):
            predictions["LDA" + str(k) + "Topic" + str(i)] = p
    #print(predictions)

    max_score = 0.0
    for ct in cluster_topics:
        if ct == "cluster_Others":
            continue
        #print(ct)
        score = predictions[ct.split('_')[-1]]
        #print(score)
        if score > max_score:
            max_score = score
            belong_cluster = ct
    #print(belong_cluster)
    if max_score < 0.3:
        belong_cluster = "cluster_Others"

    for c in clusters:
        if data[j]['PFX'] in clusters[c]['sessionPFXs']:
            true_cluster = c
            #print(c)

    if belong_cluster == 'cluster_' + true_cluster:
        correct += 1
    else:
        print(j)

print("Correct are", str(correct), "out of", str(len(wordDoc)))

# making my inference to separate clusters
cluster_separation = {}
for ct in cluster_topics:
    if ct == 'cluster_Others':
        continue
    cluster_separation[ct] = []

for j in range(len(wordDoc)):
    sequence = wordDoc[j]
    predictions = {}
    for k in range(2,max_num):
        probs = models["LDA" + str(k)].transform(sequence)[0]
        for i, p in enumerate(probs):
            predictions["LDA" + str(k) + "Topic" + str(i)] = p
    #print(predictions)

    max_score = 0.0
    for ct in cluster_topics:
        if ct == "cluster_Others":
            continue
        #print(ct)
        score = 0.0
        for c in cluster_topics[ct]:
            score += predictions[c]
        score /= len(cluster_topics[ct])
        #print(score)
        if score > max_score:
            max_score = score
            belong_cluster = ct
        #print(belong_cluster)

    cluster_separation[belong_cluster].append(data[j])


with open("inferred_clusters.json", "w") as outp:
    json.dump(cluster_separation, outp)  
