import numpy as np
import lda
import json
import csv
from collections import OrderedDict
import joblib
from sklearn.cluster import KMeans

DATA_FILE = "training_yelpchi.json" # "training_timeser.json" # "../training_adfald.json" # 
ACTIONS_FILE = "yelpchi_vocab.txt" # "timeser_vocab.txt" # "../adfald_vocabulary.txt" # 
CLUSTERS_NUM = 2 # yelpchi # 4 # timeser # 2 # adfald #

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
    
model = KMeans(n_clusters=CLUSTERS_NUM, random_state=1, verbose=0, max_iter=1500)
model.fit(wordDoc)
joblib.dump(model, "kmeans_sep_model")

kmeans_model = joblib.load("kmeans_sep_model")

# making my inference to separate clusters
cluster_separation = {}
for ct in range(CLUSTERS_NUM):
    cluster_separation["Cluster" + str(ct)] = []

for j in range(len(wordDoc)):
    sequence = wordDoc[j]
    predictions = {}
    distances = kmeans_model.transform(sequence.reshape(1,-1))[0]
    for i, d in enumerate(distances):
        predictions["Cluster" + str(i)] = d
    #print(predictions)

    min_dist = 1e+300
    for ct in cluster_separation:
        #print(ct)
        score = predictions[ct]
        #print(score)
        if score < min_dist:
            min_dist = score
            belong_cluster = ct
        #print(belong_cluster)
    cluster_separation[belong_cluster].append(data[j])

with open("kmeans_inferred_clusters.json", "w") as outp:
    json.dump(cluster_separation, outp)  