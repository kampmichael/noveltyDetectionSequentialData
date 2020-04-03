import json

TAKE_PART = 10

# training data
with open("training_hotels.json","r") as f:
    data_hotels = json.load(f)
with open("training_res.json","r") as f:
    data_res = json.load(f)

data_union = data_hotels
ind = len(data_hotels)
take_part_res = len(data_res) // TAKE_PART
for e in data_res[:take_part_res]:
    res_ex = {}
    res_ex['PFX'] = ind
    res_ex['actionsQueue'] = e['actionsQueue']
    data_union.append(res_ex)
    ind += 1

with open("training_yelpchi.json","w") as outp:
    json.dump(data_union, outp)

# testing data
with open("testing_hotels.json","r") as f:
    data_hotels = json.load(f)
with open("testing_res.json","r") as f:
    data_res = json.load(f)

data_union = data_hotels
ind = len(data_hotels)
normal_data_res = []
fake_data_res = []
for e in data_res:
    if e['label'] == 1:
        fake_data_res.append(e)
    else:
        normal_data_res.append(e)
take_part_normal_res = len(normal_data_res) // TAKE_PART
take_part_fake_res = len(fake_data_res) // TAKE_PART
for e in normal_data_res[:take_part_normal_res]:
    res_ex = {}
    res_ex['PFX'] = ind
    res_ex['actionsQueue'] = e['actionsQueue']
    res_ex['label'] = e['label']
    data_union.append(res_ex)
    ind += 1
for e in fake_data_res[:take_part_fake_res]:
    res_ex = {}
    res_ex['PFX'] = ind
    res_ex['actionsQueue'] = e['actionsQueue']
    res_ex['label'] = e['label']
    data_union.append(res_ex)
    ind += 1

with open("testing_yelpchi.json","w") as outp:
    json.dump(data_union, outp)