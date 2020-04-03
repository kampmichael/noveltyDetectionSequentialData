import numpy as np
import lda
import json
import csv
from collections import OrderedDict
import joblib
import spacy
spacy_nlp = spacy.load('en_core_web_sm')

DATA_FILE = "../training_fakenews.json" # "../training_yelpchi.json" # "training_timeser.json" # "training_smsspam.json" # "training_adfald.json" # "lss-asm-live-jsessions-anonymous.json" # "asm-data.json"
ACTIONS_FILE = "../fakenews_vocab.txt" # "../yelpchi_vocab.txt" # "timeser_vocab.txt" # "smsspam_vocab.txt" # "adfald_vocabulary.txt" # "old_amadeus_data_actions_list.txt"
# run multiple LDAs
maxNum = 9
CLEAN_STOPWORDS = True
CLEAN_VOCAB = "lda_fakenews_vocab.txt"

with open(DATA_FILE,"r") as f:
    data = json.load(f)
    
# get action sequences into one array
newData = []
for i in range(0,len(data)):
    if 'actionsQueue' not in data[i]:
        continue;
    actionsQueue = data[i]['actionsQueue']
    if len(actionsQueue)==0:
        continue
    newData.append(data[i])
print(len(newData))                

data = newData
# write down the document with corresponding "cleaned" sessions
with open("used_data_lda.json", "w") as outp:
    json.dump(newData, outp)

# read in the actions (or words) from the dictionary, so it is uniform for any other work
actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
for i, line in enumerate(file):
    actionIndex[line.rstrip("\n")] = i

if CLEAN_STOPWORDS:
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    vocab_to_use = OrderedDict()
    i = 0
    with open(CLEAN_VOCAB, "w") as outp:
        for w in actionIndex:
            if w in spacy_stopwords or w in ['-UNKN-', '-PUNCT-', '-PRON-', '-NUMBER-']:
                continue
            else:
                vocab_to_use[w] = i
                i += 1
            outp.write(w + "\n")
    actionIndex = vocab_to_use

# make bag of words from sequences
wordDoc = np.zeros((len(data),len(actionIndex),),dtype=np.int)
for i in range(0,len(data)):
    actionsQueue = data[i]['actionsQueue']
    for j in range(0,len(actionsQueue)):
        action = actionsQueue[j]
        if actionIndex.get(action) is None:
            continue
        wordIndex = actionIndex[action]
        wordDoc[i][wordIndex] = wordDoc[i][wordIndex]+1
    
#lists of probability distributions for vocabulary based on the exact topic of exact LDA model
# [9, topicsNum, vocabulary] 
topicWords = []
# each document (sequence) described in the terms of topics probability for the exact LDA model
# [9, documentsNum, topicsNum]
docTopics = []
for i in range(2,maxNum):
    model = lda.LDA(n_topics=i,n_iter=1500,random_state=1, refresh=500)
    model.fit(wordDoc)
    topicWord = model.topic_word_
    topicWords.append(topicWord)
    docTopic = model.transform(wordDoc) #model.doc_topic_
    docTopics.append(docTopic)
    joblib.dump(model, "LDA" + str(i) + "_model")
    print("round",i)
len(topicWords)

# get names of topics from corresponding LDAs
docTopicHeads = []
for i in range(2,maxNum):
    for j in range(0,i):
        st = "LDA"+str(i)+"Topic"+str(j)
        docTopicHeads.append(st)

# write actions corresponding to topics
actionList = list(actionIndex.keys())
with open("lda_topic_word_final.csv","w", newline='') as fp:
    writer = csv.writer(fp, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    actionList.append("LDA")
    writer.writerow(actionList)
    for i in range(0,len(topicWords)):
        # i + 2 will be equal to number of topics in the current LDA
        # we want to have a number indicating the amount of topics in the model that this distribution corresponds to in the last column "LDA"
        temp = np.array([[i+2]]*len(topicWords[i]))
        mergedList = np.append(topicWords[i],temp,axis=1)
        writer.writerows(mergedList)

# write documents (sessions) decomposition into topics
with open("lda_doc_topic_final.csv","w", newline='') as fp:
    writer = csv.writer(fp, delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    writer.writerow(docTopicHeads)
    mergedList = docTopics[0]
    for i in range(1,len(docTopics)):
        mergedList = np.append(mergedList,docTopics[i],axis=1)
    writer.writerows(mergedList)
