import json
import pandas as pd
from collections import Counter
path = "src/utils/categories.json"

with open(path) as f:
    categories = json.load(f)
    
category_list = [category for category in categories.values()]

counts = Counter(category_list)

counts_df = pd.DataFrame.from_dict(counts, orient='index').reset_index().rename(columns={'index':'category', 0:'count'})
counts_df.to_csv('src/utils/category_counts.csv', index=False)



