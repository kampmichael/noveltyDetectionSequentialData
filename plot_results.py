import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import cycle
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import os

def calcSensAndSpec(cm):
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    tp = cm[1,1]
    print(cm)
    TPR = float(tp)/float(tp + fn)
    TNR = float(tn)/float(tn + fp)
    return TPR, TNR

viridis = cm.get_cmap('viridis', 256)
colors = viridis(np.linspace(0.2, 0.8, 3))

data_path = "yelpchi_data" # "smsspam" # "fake_news" # "adfald_data/emb128_hid512" # "timeseries_data_2" # "genome_dataset" # 
DATA = "Yelp chi reviews" # "ADFA LD" # "Fakenews" # "Timeseries" # 
# Binarize the output
test_labels = np.load(os.path.join(data_path, 
    "test_labels.npy")) # pure_LDA_test_labels kmeans_test_labels 
cluster_predictions = np.load(os.path.join(data_path, 
    "cluster_perplexities.npy")) # pure_LDA_cluster_perplexities  kmeans_cluster_perplexities
global_predictions = np.load(os.path.join(data_path, 
    "global_perplexities.npy")) # pure_LDA_global_perplexities  kmeans_global_perplexities

# Compute ROC curve and ROC area
fpr = {}
tpr = {}
roc_auc = {}
fpr['clusters'], tpr['clusters'], cluster_thresholds = roc_curve(test_labels, cluster_predictions)
roc_auc['clusters'] = auc(fpr['clusters'], tpr['clusters'])
fpr['global'], tpr['global'], global_thresholds = roc_curve(test_labels, global_predictions)
roc_auc['global'] = auc(fpr['global'], tpr['global'])

cl_thresh = cluster_thresholds[len(cluster_thresholds)//2]
gl_thresh = global_thresholds[len(global_thresholds)//2]
cm = confusion_matrix(test_labels, [1 if p > gl_thresh else 0 for p in global_predictions])
print("Global", calcSensAndSpec(cm))
cm = confusion_matrix(test_labels, [1 if p > cl_thresh else 0 for p in cluster_predictions])
print("Clusters", calcSensAndSpec(cm))

plt.figure()
lw = 2
plt.plot(fpr['clusters'], tpr['clusters'], lw=lw, color = colors[0],
	label='Clusters: ROC curve (area = %0.4f)' % roc_auc['clusters'])
plt.plot(fpr['global'], tpr['global'], lw=lw, color = colors[1], 
	label='Global: ROC curve (area = %0.4f)' % roc_auc['global'])
plt.plot([0, 1], [0, 1], color='black', alpha=0.5, lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(DATA)
plt.legend(loc="lower right")
plt.show()