import spacy
import re
import numpy as np
import operator
sp = spacy.load('en_core_web_sm')

PUNCTS = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/',
 '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}',
  '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â',
   '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
    '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕',
     '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
      '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚',
       '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def preprocess_text(t):
    clean_t = []
    for w in sp(t):
        clean_t.append(w.lemma_)

    temp = []
    for w in clean_t:
        if bool(re.search(r'\d', w)):
            temp.append('-NUMBER-')
        elif w in PUNCTS:
            temp.append('-PUNCT-')
        else:
            temp.append(w)
    clean_t = temp
    return np.array(clean_t)


hotel_data = open("output_review_yelpHotelData_NRYRcleaned.txt", "r").read().split("\n")[:-1]
hotel_labels = []
for l in open("output_meta_yelpHotelData_NRYRcleaned.txt", "r").read().split("\n")[:-1]:
    hotel_labels.append(int(l.split()[4] == 'Y'))
print("Hotel data size", len(hotel_data))
res_data = open("output_review_yelpResData_NRYRcleaned.txt", "r").read().split("\n")[:-1]
res_labels = []
for l in open("output_meta_yelpResData_NRYRcleaned.txt", "r").read().split("\n")[:-1]:
    res_labels.append(int(l.split()[4] == 'Y'))
print("Restaurant data size", len(res_data))

vocabulary = {}

tr_hotel_data = []
hotel_normal_count = 0
hotel_fraud_count = 0
for i, d in enumerate(hotel_data):
    tr_hotel_data.append(preprocess_text(d))
    for w in tr_hotel_data[-1]:
        if w in set(list(vocabulary.keys())):
            vocabulary[w] += 1
        else:
            vocabulary[w] = 1
    if hotel_labels[i] == 1:
        hotel_fraud_count += 1
    else:
        hotel_normal_count += 1
    if i%1000 == 0:
        print("On", i, "th hotel example")
# 5076 and 778
print("Hotel normal", hotel_normal_count, "Hotel fraud", hotel_fraud_count)
np.save("preprocessed_hotel_data", tr_hotel_data)
np.save("hotel_labels", hotel_labels)

tr_res_data = []
res_normal_count = 0
res_fraud_count = 0
for i, d in enumerate(res_data):
    tr_res_data.append(preprocess_text(d))
    for w in tr_res_data[-1]:
        if w in set(list(vocabulary.keys())):
            vocabulary[w] += 1
        else:
            vocabulary[w] = 1
    if res_labels[i] == 1:
        res_fraud_count += 1
    else:
        res_normal_count += 1
    if i%1000 == 0:
        print("On", i, "th restaurant example")
# Restaurant normal 53400 Restaurant fraud 8141
print("Restaurant normal", res_normal_count, "Restaurant fraud", res_fraud_count)
np.save("preprocessed_res_data", tr_res_data)
np.save("res_labels", res_labels)

sorted_v = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
np.save("full_vocabulary_sorted", sorted_v)
