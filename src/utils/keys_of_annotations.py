import json 
import pandas as pd 

path = "./src/data/ai2d/annotations/4.png.json"

with open(path) as f:
    data = json.load(f)

data_keys = data.keys()
data_subkeys = [data[key].keys() for key in data_keys]
print(data_subkeys)