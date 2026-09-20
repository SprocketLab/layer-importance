import json

with open("compressed_data.json") as infile:
    data = json.load(infile)

for k, v in data.items():
    #k = k.split(" - ")[0]
    v = sum(v) / len(v)
    print(k, v)
    
