import json
from collections import OrderedDict

DATA_FILE = "all_text_sessions.json"
ACTIONS_FILE = "sc.txt"
ATTACKS_FILE = "attack_text_sessions.json"

with open(DATA_FILE,"r") as f:
    data = json.load(f)
    print(len(data))

training_length = int(len(data) * 0.8)

with open("training_adfald.json", "w") as outp:
    training_data = []
    for i in range(training_length):
        training_data.append({})
        training_data[-1]['PFX'] = data[i]['id']
        training_data[-1]['actionsQueue'] = data[i]['actions']
    json.dump(training_data, outp)

with open("testing_adfald.json","w") as outp:
    testing_data = []
    for i in range(training_length, len(data)):
        testing_data.append({})
        testing_data[-1]['PFX'] = data[i]['id']
        testing_data[-1]['actionsQueue'] = data[i]['actions']
        testing_data[-1]['label'] = 0
    attack_data = json.load(open(ATTACKS_FILE, "r"))
    print("attack data", len(attack_data))
    for i in range(len(attack_data)):
        testing_data.append({})
        testing_data[-1]['PFX'] = attack_data[i]['id']
        testing_data[-1]['actionsQueue'] = attack_data[i]['actions']
        testing_data[-1]['label'] = 1
    json.dump(testing_data, outp)

actionIndex = OrderedDict()
file = open(ACTIONS_FILE)
    for i, line in enumerate(file):
        actionIndex[line.rstrip("\n").split("\t")[1]] = i

with open("adfald_vocabulary.txt", "w") as outp:
    for k in list(actionIndex.keys()):
        outp.write(k + "\n")


