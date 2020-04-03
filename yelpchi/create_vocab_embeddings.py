import json
from collections import OrderedDict
import numpy as np

vocab = open("../yelpchi_vocab.txt", "r").readlines()

EMB_LEN = 100

glove_vocab = {}
with open("../../glove.6B.100d.txt", 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)
        glove_vocab[word] = vect

# create vocabulary embeddings array
vocabulary = []
vocabulary_embeddings = []
print("Print not found in GLOVE words")
for p in vocab:
    word = p.replace("\n", "")
    if glove_vocab.get(word) is None:
        print(repr(word))
        vocabulary_embeddings.append(np.random.rand(EMB_LEN))
    else:
        vocabulary_embeddings.append(glove_vocab[word])
np.save("yelpchi_vocab_emb", vocabulary_embeddings)