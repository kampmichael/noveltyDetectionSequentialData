import numpy as np
import lda
import json
import csv
from collections import OrderedDict
import joblib

DATA_FILE = "training_timeser.json" # "training_adfald.json" # "training_yelpchi.json" # 
ACTIONS_FILE = "timeser_vocab.txt" # "adfald_vocabulary.txt" # "yelpchi_vocab.txt" # 
TOPICS_NUM = 13 # timeser2 # 2 # adfald #

with open(DATA_FILE,"r") as f:
    data = json.load(f)
    
# get action sequences into one array
sequences = []
for i in range(0,len(data)):
    actionsQueue = data[i]['actionsQueue']
    sequences.append(actionsQueue)
print("Size of sequences", len(sequences))

# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i

# make bag of words from sequences
wordDoc = np.zeros((len(sequences),len(actionIndex),),dtype=np.int)
for i in range(0,len(data)):
    actionsQueue = data[i]['actionsQueue']
    for j in range(0,len(actionsQueue)):
        action = actionsQueue[j]
        wordIndex = actionIndex[action]
        wordDoc[i][wordIndex] = wordDoc[i][wordIndex]+1
    
model = lda.LDA(n_topics=TOPICS_NUM, n_iter=1500, random_state=1, refresh=500)
model.fit(wordDoc)
joblib.dump(model, "pure_LDA_sep_model")

lda_model = joblib.load("pure_LDA_sep_model")

# making my inference to separate clusters
cluster_separation = {}
for ct in range(TOPICS_NUM):
    cluster_separation["Topic" + str(ct)] = []

for j in range(len(wordDoc)):
    sequence = wordDoc[j]
    predictions = {}
    probs = lda_model.transform(sequence)[0]
    for i, p in enumerate(probs):
        predictions["Topic" + str(i)] = p
    #print(predictions)

    max_score = 0.0
    for ct in cluster_separation:
        #print(ct)
        score = predictions[ct]
        #print(score)
        if score > max_score:
            max_score = score
            belong_cluster = ct
        #print(belong_cluster)
    cluster_separation[belong_cluster].append(data[j])

with open("pure_LDA_inferred_clusters.json", "w") as outp:
    json.dump(cluster_separation, outp)  