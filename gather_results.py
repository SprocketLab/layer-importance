import json
import ast
import sys
import os
from collections import defaultdict

#runid = sys.argv[1]

data = []
for filename in os.listdir("results"):
#for filename in os.listdir("logs"):
    #if not runid in filename: continue
    #with open("logs/" + filename) as infile:
    with open("results/" + filename) as infile:
        data += [ast.literal_eval(line) for line in infile.read().strip().split("\n") if line.startswith("{'args")]

out_data = defaultdict(lambda: defaultdict(list))
for elem in data:
    elem = elem['curbes']
    k = list(elem.keys())[0]
    elem = elem[k][0]
    index =  '-'.join(elem['layers'].split(","))
    index += " - dim%d" % elem['d']
    if elem['window'] == 'full':
        index += " - window%d" % elem['length']
    else:
        index += " - window%d" % elem['window']
    index += " - state%d" % elem['state_dim']
    index += " - mem%d" % elem['memory']
    index += " - param%d" % elem['params']
    out_data[index][float(elem['lr'])].append(float(elem['best_final_acc']))

out_data2 = defaultdict(list)
for k in out_data.keys():
    for k2 in out_data[k].keys():
        out_data2[k].append((sum(out_data[k][k2]) / len(out_data[k][k2]), k2))

lrs = {}
for k in out_data2.keys():
    out_data2[k].sort(key = lambda x: -x[0])
    lrs[k] = out_data2[k][0][1]

out_data3 = defaultdict(list)
for k in out_data.keys():
    out_data3[k] = out_data[k][lrs[k]]

with open('compressed_data.json', 'w') as outfile:
    json.dump(out_data3, outfile, indent=4)
