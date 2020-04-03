import csv
import numpy as np
import json
import os

WINDOW_SIZE = 40
STEP_SIZE = 40

series_data = {}
for f in os.listdir('.'):
    if 'ec2_cpu_utilization' in f:
        with open(f, "r") as csvfile:
            reader = csv.reader(csvfile)
            key = f.replace('.csv','')
            series_data[key] = []
            i = 0
            for row in reader:
                if not i == 0:
                    series_data[key].append(row)
                i += 1

windows = {}
for k in series_data:
    windows[k] = []
    for i in range(0, len(series_data[k]), STEP_SIZE):
        if len(series_data[k][i:i+WINDOW_SIZE]) == WINDOW_SIZE:
            windows[k].append(series_data[k][i:i+WINDOW_SIZE])

anomalies = {}
d = json.load(open("combined_labels.json", "r"))
for k in d:
    if 'ec2_cpu_utilization' in k:
        anomalies[k.split('/')[1].replace('.csv','')] = set(d[k])

normal_seqs = []
anomaly_seqs = []
for k in windows:
    for w in windows[k]:
        window_timerange = set(np.array(w)[:,0])
        float_values = [float(v) for v in np.array(w)[:,1]]
        if window_timerange & anomalies[k]:
            anomaly_seqs.append(float_values)
        else:
            normal_seqs.append(float_values)
normal_count = len(normal_seqs)
anomaly_count = len(anomaly_seqs)
print("Normal count", normal_count, "Anomaly count", anomaly_count)

gl_min = np.array(anomaly_seqs).min()
if np.array(normal_seqs).min() < gl_min:
    gl_min = np.array(normal_seqs).min()
gl_max = np.array(anomaly_seqs).max()
if np.array(normal_seqs).max() > gl_max:
    gl_max = np.array(normal_seqs).max()
print("Min", gl_min, "Max", gl_max)
bins = np.arange(0, 100, 0.1)

training_length = int(normal_count * 0.8)
ind = 0
with open("training_timeser.json", "w") as outp:
    training_data = []
    for i in range(training_length):
        training_data.append({})
        training_data[-1]['PFX'] = ind
        training_data[-1]['actionsQueue'] = [str(e) for e in np.digitize(normal_seqs[i],bins)] 
        ind += 1
    json.dump(training_data, outp)

testing_length = normal_count - training_length
with open("testing_timeser.json", "w") as outp:
    testing_data = []
    for i in range(training_length, normal_count):
        testing_data.append({})
        testing_data[-1]['PFX'] = ind
        testing_data[-1]['actionsQueue'] = [str(e) for e in np.digitize(normal_seqs[i],bins)]
        testing_data[-1]['label'] = 0
        ind += 1
    for seq in anomaly_seqs:
        testing_data.append({})
        testing_data[-1]['PFX'] = ind
        testing_data[-1]['actionsQueue'] = [str(e) for e in np.digitize(seq,bins)]
        testing_data[-1]['label'] = 1
        ind += 1
    json.dump(testing_data, outp)

with open("timeser_vocab.txt", "w") as outp:
    for e in range(len(bins) + 1):
        outp.write(str(e) + "\n")