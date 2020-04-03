from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json
from collections import OrderedDict
import lda

DATA_FILE = "training_fakenews.json" # "yelpchi_data/training_yelpchi.json" # "timeseries_data_2/training_timeser.json" # "adfald_data/training_adfald.json" # "training_yelpchi.json" # 
ACTIONS_FILE = "fakenews_vocab.txt" # "yelpchi_data/yelpchi_vocab.txt" # "timeseries_data_2/timeser_vocab.txt" # "adfald_data/adfald_vocabulary.txt" # "yelpchi_vocab.txt" #

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

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots(1, 1)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    ax.set_xlim([-1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(wordDoc) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    #clusterer = KMeans(n_clusters=n_clusters, random_state=1, verbose=0, max_iter=1500)
    #cluster_labels = clusterer.fit_predict(wordDoc)
    clusterer = lda.LDA(n_topics=n_clusters, n_iter=1500, random_state=1, refresh=1500)
    clusterer.fit(wordDoc)
    cluster_labels = []
    for d in wordDoc:
        pred = clusterer.transform(d)[0]
        cluster_labels.append(pred.argmax())

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(wordDoc, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(wordDoc, cluster_labels)

    y_lower = 10
    '''
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()
'''
# LDA for yelpchi
#2 0.0097
#3 -0.036
#4 -0.054
#5 -0.16
#6 -0.12
#7 -0.1
#8 -0.11
#9 -0.14
#10 -0.15

# LDA for timeseries_2
#2 = 0.15
#3 0.09
#4 = 0.20
#5 0.15
#6 0.20
#7 0.23
#8 0.2446
#9 0.31
#10 0.32
#11 0.32
#12 0.35
#13 0.37
#14 0.15
#15 0.24

# LDA for adfald
#For n_clusters = 2 The average silhouette_score is : 0.28620387811232045
#For n_clusters = 3 The average silhouette_score is : 0.24104071068957186
#For n_clusters = 4 The average silhouette_score is : 0.2605156041242335
#For n_clusters = 5 The average silhouette_score is : 0.2627979516139205
#For n_clusters = 6 The average silhouette_score is : 0.11590239492536676
#For n_clusters = 7 The average silhouette_score is : 0.06946298586779934
#For n_clusters = 8 The average silhouette_score is : 0.019055868493228805
#For n_clusters = 9 The average silhouette_score is : 0.04389023337428432