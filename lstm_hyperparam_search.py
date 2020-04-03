import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from collections import OrderedDict
import numpy as np
import math

torch.manual_seed(1)
ACTIONS_FILE = "../yelpchi_vocab.txt" # "fakenews_vocab.txt" # "adfald_vocabulary.txt" # "timeser_vocab.txt" # "smsspam_vocab.txt" 
# 
DATA_FILE = "../training_yelpchi.json" # "training_fakenews.json" # "../training_adfald.json" # "training_timeser.json" # "training_smsspam.json" 
# 

BATCH = 64 # 8 # timeser # 
PAD_IDX = 6024 # yelpchi # 5144 # fakenews # 1001 # timeser # 914 # smsspam # 341 # adfald # 
USE_EMB = False
EMBEDDING_DIM = 100
MAX_LEN = 2000

if USE_EMB:
    weights_matrix = np.load("fakenews_vocab_emb.npy")
    weights_matrix = np.append(weights_matrix, [np.random.rand(EMBEDDING_DIM)], axis = 0)

class LSTMNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = PAD_IDX)
        if USE_EMB:
            self.word_embeddings.load_state_dict({'weight': torch.tensor(weights_matrix)})

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
        if math.isinf(product * (1.0/p)):
            product = 1e+300
        else:
            product *= (1.0/p)
        count += 1
    return product**(1.0/count)
    
# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i
actionIndex['PAD'] = PAD_IDX
print(actionIndex)

with open(DATA_FILE, "r") as f:
    data = json.load(f)
    print("Amount of clusters", len(data))

global_training_data = []
global_max_len = 0
training_sequence_ids = []
for s in data:
    training_sequence_ids.append(s['PFX'])
    target = s['actionsQueue'][1:]
    target.append('PAD')
    global_training_data.append((s['actionsQueue'], target))
    if len(s['actionsQueue']) > global_max_len:
        global_max_len = len(s['actionsQueue'])

if global_max_len > MAX_LEN:
    global_max_len = MAX_LEN
    print("Shrinked the max len")

print("Global training data size", len(global_training_data))
print("Global maximal length of sequence", global_max_len)
training_tensors = []
for s,t in global_training_data:
    training_tensors.append((prepare_sequence(s, actionIndex, global_max_len), 
        prepare_sequence(t, actionIndex, global_max_len)))
print(len(training_tensors))

validation_len = int(len(training_tensors) * 0.1)
validation_ids = training_sequence_ids[:validation_len]
# use later in the training to check if the sequence should be used or not
# do not use for training the sequences with id from validation_ids
np.save("validation_ids", validation_ids)
validation_set = training_tensors[:validation_len]
training_set = training_tensors[validation_len:]

res = []
for emb_dim in [32, 64, 128, 256, 512]:
    if not emb_dim == 32 and USE_EMB:
        break
    if emb_dim == 32 and USE_EMB:
        emb_dim = EMBEDDING_DIM
    for hid_dim in [32, 64, 128, 256, 512]:
        print("Check emb_dim", emb_dim, "hid_dim", hid_dim)
        model = LSTMNet(emb_dim, hid_dim, len(actionIndex), len(actionIndex)).cuda()
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(10):
            #print("epoch", epoch)
            batch_seq = []
            batch_t = []
            np.random.shuffle(training_set)
            for sequence, targets in training_set:
                batch_seq.append(sequence)
                batch_t.append(targets)
                if len(batch_seq) == BATCH:
                    model.zero_grad()
                    tag_scores = model(torch.stack(batch_seq, dim=0).cuda())
                    loss = loss_function(tag_scores, torch.stack(batch_t, dim=0).cuda())
                    loss.backward()
                    optimizer.step()
                    batch_seq = []
                    batch_t = []

        model.eval()

        av_perplexity = 0
        av_loss = 0
        for sequence, targets in validation_set:
            tag_scores = model(sequence.cuda().view(1,-1))
            loss = loss_function(tag_scores, targets.cuda().view(1,-1))
            av_loss += loss.data.cpu().item()
            tag_scores = tag_scores.squeeze().exp().data.cpu().numpy()
            probs = []
            for i,t in enumerate(targets.data.numpy()):
                if t != PAD_IDX:
                    probs.append(tag_scores[:,i][t])
            av_perplexity += perplexity(probs)
        print("Validation loss", av_loss/len(validation_set))
        print("Validation perplexity", av_perplexity/len(validation_set))

        res.append("emb_dim" + str(emb_dim) + ",hid_dim" + str(hid_dim) + 
            ",val_loss" + str(av_loss/len(validation_set)) + ",val_perpl" + str(av_perplexity/len(validation_set)))
        
np.save("hyperparameters_eval", res)