import numpy as np
import json
from sklearn.metrics import roc_auc_score

CLUSTERS_SEPARATION_FILE = "inferred_clusters.json" # "inferred_clusters.json" #
PER_CLUSTER_RESULTS = "per_cluster_perplexities.json" # "per_cluster_perplexities.json" #

with open(CLUSTERS_SEPARATION_FILE, "r") as f:
    data = json.load(f)
    print("-----Amount of clusters", len(data))

validation_set_ids = np.load(open("../validation_ids.npy", "rb"))

with open(PER_CLUSTER_RESULTS, "r") as f:
    results = json.load(f)

# each sequence has [label, cluster_score, global_score]
for cl in results:
    print("-----", cl)
    training_data_count = 0
    for s in data[cl]:
        if s['PFX'] in validation_set_ids:
            continue
        training_data_count += 1
    print("Training data size", training_data_count)
    cl_res = np.array(results[cl])
    try:
        print("Cluster AUC", roc_auc_score(cl_res[:,0], cl_res[:,1]))
        print("Global AUC", roc_auc_score(cl_res[:,0], cl_res[:,2]))
    except ValueError:
        print("Cluster without one class")
