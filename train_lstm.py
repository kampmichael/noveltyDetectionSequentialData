import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from collections import OrderedDict
import numpy as np
import math

torch.manual_seed(42)
ACTIONS_FILE = "../fakenews_vocab.txt" # "../yelpchi_vocab.txt" # 'timeser_vocab.txt' # "smsspam_vocab.txt" # "genom_vocab.txt" 
# "adfald_vocabulary.txt" # 
CLUSTERS_SEPARATION_FILE = "inferred_clusters.json" # "kmeans_inferred_clusters.json" #
# "../full_inferred_clusters.json" # 
TRAIN_GLOBAL = True

EMBEDDING_DIM = 256 # fakenews2 # 256 # yelpchi # 100 # fakenews # 128 # timeser2 # 256 # timeser # 64 # smsspam 
# 128 # genom # 128 # adfald # 
USE_EMB = False
HIDDEN_DIM = 512 # fakenews2 # 512 # yelpchi # 512 # fakenews # 64 # timeser2 # 64 # timeser # 128 # smsspam 
# 128 # genom # 512 # adfald # 
BATCH = 32 # 8 # 
PAD_IDX = 5144 # fakenews # 6024 # yelpchi # 1001 # timeser2 # 541 # timeser # 914 # smsspam 
# 5 # genom # 341 # adfald # 
EPOCHS_NUM = 800 # 500 for adfald
print_period_epochs = 100
MAX_LEN = 2000

if USE_EMB:
    weights_matrix = np.load("../fakenews_vocab_emb.npy")
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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)

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

    
# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i
actionIndex['PAD'] = PAD_IDX

with open(CLUSTERS_SEPARATION_FILE, "r") as f:
    data = json.load(f)
    print("Amount of clusters", len(data))

validation_set_ids = np.load(open("../validation_ids.npy", "rb"))

global_training_data = []
global_max_len = 0
for cl in data:
    if cl == 'cluster_Others':
        continue
    print(cl)

    training_data = []
    max_len = 0
    for s in data[cl]:
        if s['PFX'] in validation_set_ids:
            print("excluded validation example")
            continue
        target = s['actionsQueue'][1:]
        target.append('PAD')
        training_data.append((s['actionsQueue'], target))
        global_training_data.append((s['actionsQueue'], target))
        if len(s['actionsQueue']) > max_len:
            max_len = len(s['actionsQueue'])
        if max_len > global_max_len:
            global_max_len = max_len
    if max_len > MAX_LEN:
        max_len = MAX_LEN
        print("Shrinked the max len")
    print("Training data size", len(training_data))
    print("Maximal length of sequence", max_len)
    training_tensors = []
    for s,t in training_data:
        training_tensors.append((prepare_sequence(s, actionIndex, max_len), prepare_sequence(t, actionIndex, max_len)))
    print(len(training_tensors))

    model = LSTMNet(EMBEDDING_DIM, HIDDEN_DIM, len(actionIndex), len(actionIndex)).cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters()) #optim.SGD(model.parameters(), lr=0.01)

    ## See what the scores are before training
    ## Note that element i,j of the output is the score for tag j for word i.
    ## Here we don't need to train, so the code is wrapped in torch.no_grad()
    #with torch.no_grad():
    #    batch = []
    #    for e in training_tensors[:BATCH]:
    #        batch.append(e[0])
    #    tag_scores = model(torch.stack(batch, dim=0).cuda())
    #    print(tag_scores.data.cpu().numpy().argmax(axis=1))
    #    print(batch)

    for epoch in range(EPOCHS_NUM):
        if len(training_tensors) == 0:
            print("No training data")
            break
        epoch_loss = 0.0
        batch_seq = []
        batch_t = []
        np.random.shuffle(training_tensors)
        for sequence, targets in training_tensors:
            batch_seq.append(sequence)
            batch_t.append(targets)
            if len(batch_seq) == BATCH:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 3. Run our forward pass.
                tag_scores = model(torch.stack(batch_seq, dim=0).cuda())

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, torch.stack(batch_t, dim=0).cuda())
                epoch_loss += loss.data.cpu().item()
                loss.backward()
                optimizer.step()
                batch_seq = []
                batch_t = []

        if epoch%print_period_epochs == 0:
            print("Epoch", epoch)
            print("Training loss", epoch_loss/len(training_tensors))

    ## See what the scores are after training
    #with torch.no_grad():
    #    batch = []
    #    for e in training_tensors[:BATCH]:
    #        batch.append(e[0])
    #    tag_scores = model(torch.stack(batch, dim=0).cuda())
    #    print(tag_scores.data.cpu().numpy().argmax(axis=1))
    #    print(batch)

    torch.save(model.state_dict(), cl + "lstm_model")


if global_max_len > MAX_LEN:
    global_max_len = MAX_LEN
    print("Shrinked the max len")
print("Global training data size", len(global_training_data))
print("Global maximal length of sequence", global_max_len)
if TRAIN_GLOBAL:
    training_tensors = []
    for s,t in global_training_data:
        training_tensors.append((prepare_sequence(s, actionIndex, global_max_len), prepare_sequence(t, actionIndex, global_max_len)))
    print(len(training_tensors))

    model = LSTMNet(EMBEDDING_DIM, HIDDEN_DIM, len(actionIndex), len(actionIndex)).cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters()) #optim.SGD(model.parameters(), lr=0.01)

    ## See what the scores are before training
    ## Note that element i,j of the output is the score for tag j for word i.
    ## Here we don't need to train, so the code is wrapped in torch.no_grad()
    #with torch.no_grad():
    #    batch = []
    #    for e in training_tensors[:BATCH]:
    #        batch.append(e[0])
    #    tag_scores = model(torch.stack(batch, dim=0).cuda())
    #    print(tag_scores.data.cpu().numpy().argmax(axis=1))
    #    print(batch)

    for epoch in range(EPOCHS_NUM):
        epoch_loss = 0.0
        batch_seq = []
        batch_t = []
        np.random.shuffle(training_tensors)
        for sequence, targets in training_tensors:
            batch_seq.append(sequence)
            batch_t.append(targets)
            if len(batch_seq) == BATCH:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 3. Run our forward pass.
                tag_scores = model(torch.stack(batch_seq, dim=0).cuda())

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, torch.stack(batch_t, dim=0).cuda())
                epoch_loss += loss.data.cpu().item()
                loss.backward()
                optimizer.step()
                batch_seq = []
                batch_t = []

        if epoch%print_period_epochs == 0:
            print("Epoch", epoch)
            print("Training loss", epoch_loss/len(training_tensors))

    ## See what the scores are after training
    #with torch.no_grad():
    #    batch = []
    #    for e in training_tensors[:BATCH]:
    #        batch.append(e[0])
    #    tag_scores = model(torch.stack(batch, dim=0).cuda())
    #    print(tag_scores.data.cpu().numpy().argmax(axis=1))
    #    print(batch)

    torch.save(model.state_dict(), "global_lstm_model")

# training accuracy calculation
# the smaller perplexity, the better sequence is described by the model
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

for cl in data:
    if cl == 'cluster_Others':
        continue

    training_data = []
    for s in data[cl]:
        target = s['actionsQueue'][1:]
        target.append('PAD')
        training_data.append((s['actionsQueue'], target))
    print("-----Cluster", cl, "size is", len(training_data))
    training_tensors = []
    for s,t in training_data:
        training_tensors.append((prepare_sequence(s, actionIndex, len(s)), prepare_sequence(t, actionIndex, len(t))))

    model = LSTMNet(EMBEDDING_DIM, HIDDEN_DIM, len(actionIndex), len(actionIndex)).cuda()
    model.load_state_dict(torch.load(cl + "lstm_model"))
    model.eval()

    av_perplexity = 0
    av_loss = 0
    for sequence, targets in training_tensors:
        tag_scores = model(sequence.cuda().view(1,-1))
        loss = loss_function(tag_scores, targets.cuda().view(1,-1))
        av_loss += loss.data.cpu().item()
        tag_scores = tag_scores.squeeze().exp().data.cpu().numpy()
        probs = []
        for i,t in enumerate(targets.data.numpy()):
            #print("-------------", t)
            #print("max probability", tag_scores[:,i].max())
            #print("prediction", tag_scores[:,i].argmax())
            #print("probability of target", tag_scores[:,i][t])
            if t != PAD_IDX:
                probs.append(tag_scores[:,i][t])
        av_perplexity += perplexity(probs)
    print("Training loss", av_loss/len(training_tensors))
    print("Training perplexity", av_perplexity/len(training_tensors))

if TRAIN_GLOBAL:
    training_tensors = []
    for s,t in global_training_data:
        training_tensors.append((prepare_sequence(s, actionIndex, len(s)), prepare_sequence(t, actionIndex, len(t))))
    print("-----All clusters size is", len(training_tensors))

    model = LSTMNet(EMBEDDING_DIM, HIDDEN_DIM, len(actionIndex), len(actionIndex)).cuda()
    model.load_state_dict(torch.load("global_lstm_model"))
    model.eval()

    av_perplexity = 0
    av_loss = 0
    for sequence, targets in training_tensors:
        tag_scores = model(sequence.cuda().view(1,-1))
        loss = loss_function(tag_scores, targets.cuda().view(1,-1))
        av_loss += loss.data.cpu().item()
        tag_scores = tag_scores.squeeze().exp().data.cpu().numpy()
        probs = []
        for i,t in enumerate(targets.data.numpy()):
            #print("-------------", t)
            #print("max probability", tag_scores[:,i].max())
            #print("prediction", tag_scores[:,i].argmax())
            #print("probability of target", tag_scores[:,i][t])
            if t != PAD_IDX:
                probs.append(tag_scores[:,i][t])
        av_perplexity += perplexity(probs)
    print("Training loss", av_loss/len(training_tensors))
    print("Training perplexity", av_perplexity/len(training_tensors))
