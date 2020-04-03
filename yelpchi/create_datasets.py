import json
from collections import OrderedDict
import numpy as np 

sorted_v = np.load("full_vocabulary_sorted.npy")
tr_res_data = np.load("preprocessed_res_data.npy", allow_pickle=True)
res_labels = np.load("res_labels.npy")
tr_hotel_data = np.load("preprocessed_hotel_data.npy", allow_pickle=True)
hotel_labels = np.load("hotel_labels.npy")

UNK = "-UNKN-"
FREQ_THRESH = 40 # yelpchi #

# helps to get an impression of how often the words are
freqs = np.array([int(p[1]) for p in sorted_v])
np.percentile(freqs, 50)
np.percentile(freqs, 90)
# create vocabulary of most often words and add one more UNKN
vocabulary = []
for p in sorted_v:
    if int(p[1]) >= FREQ_THRESH:
        vocabulary.append(p[0])
vocabulary.append(UNK)
print("Vocabulary is of size", len(vocabulary))
with open("yelpchi_vocab.txt", "w") as outp:
    for w in vocabulary:
        outp.write(w + "\n")
words_set = set(vocabulary)

# make training and testing sets from hotel data
normal_seqs_hotel = []
attack_seqs_hotel = []
for i, s in enumerate(tr_hotel_data):
    if hotel_labels[i] == 1:
        attack_seqs_hotel.append(s)
    else:
        normal_seqs_hotel.append(s)
training_length_h = int(len(normal_seqs_hotel) * 0.8)
ind_h = 0
with open("training_hotels.json", "w") as outp:
    training_data_h = []
    for i in range(training_length_h):
        training_data_h.append({})
        training_data_h[-1]['PFX'] = ind_h
        seq = []
        for w in normal_seqs_hotel[i]:
            if w in words_set:
                seq.append(w)
            else:
                seq.append(UNK)
        training_data_h[-1]['actionsQueue'] = seq
        ind_h += 1
    json.dump(training_data_h, outp)

with open("testing_hotels.json", "w") as outp:
    testing_data_h = []
    for i in range(training_length_h, len(normal_seqs_hotel)):
        testing_data_h.append({})
        testing_data_h[-1]['PFX'] = ind_h
        seq = []
        for w in normal_seqs_hotel[i]:
            if w in words_set:
                seq.append(w)
            else:
                seq.append(UNK)
        testing_data_h[-1]['actionsQueue'] = seq
        testing_data_h[-1]['label'] = 0
        ind_h += 1
    for s in attack_seqs_hotel:
        testing_data_h.append({})
        testing_data_h[-1]['PFX'] = ind_h
        seq = []
        for w in s:
            if w in words_set:
                seq.append(w)
            else:
                seq.append(UNK)
        testing_data_h[-1]['actionsQueue'] = seq
        testing_data_h[-1]['label'] = 1
        ind_h += 1
    json.dump(testing_data_h, outp)

# make training and testing sets from restaurant data
normal_seqs_res = []
attack_seqs_res = []
for i, s in enumerate(tr_res_data):
    if res_labels[i] == 1:
        attack_seqs_res.append(s)
    else:
        normal_seqs_res.append(s)
training_length_r = int(len(normal_seqs_res) * 0.8)
ind_r = 0
with open("training_res.json", "w") as outp:
    training_data_r = []
    for i in range(training_length_r):
        training_data_r.append({})
        training_data_r[-1]['PFX'] = ind_r
        seq = []
        for w in normal_seqs_res[i]:
            if w in words_set:
                seq.append(w)
            else:
                seq.append(UNK)
        training_data_r[-1]['actionsQueue'] = seq
        ind_r += 1
    json.dump(training_data_r, outp)

with open("testing_res.json", "w") as outp:
    testing_data_r = []
    for i in range(training_length_r, len(normal_seqs_res)):
        testing_data_r.append({})
        testing_data_r[-1]['PFX'] = ind_r
        seq = []
        for w in normal_seqs_res[i]:
            if w in words_set:
                seq.append(w)
            else:
                seq.append(UNK)
        testing_data_r[-1]['actionsQueue'] = seq
        testing_data_r[-1]['label'] = 0
        ind_r += 1
    for s in attack_seqs_res:
        testing_data_r.append({})
        testing_data_r[-1]['PFX'] = ind_r
        seq = []
        for w in s:
            if w in words_set:
                seq.append(w)
            else:
                seq.append(UNK)
        testing_data_r[-1]['actionsQueue'] = seq
        testing_data_r[-1]['label'] = 1
        ind_r += 1
    json.dump(testing_data_r, outp)
