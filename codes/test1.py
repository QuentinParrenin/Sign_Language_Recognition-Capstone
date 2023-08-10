import pyarrow.parquet as pq
import pandas as pd
import os
import json
import numpy as np
# file_id = '1019715464'
# file_path = '/home/dragon_warrior/Documents/ASL/data/{}.parquet'.format(file_id)
# meta_csv_path = os.path.abspath('../data')
# csv_path = os.path.abspath('../data/csv_files')

# table = pd.read_parquet(file_path)
# print(table.head())
# print(type(table))

# with open ("../index/character_to_prediction_index.json", "r") as f:
#     character_map = json.load(f)
#     rev_character_map = {j:i for i,j in character_map.items()}
#     print(rev_character_map)


# predictions = np.array([[.8,0.03,0.17],
#                        [.01,0.82,0.17],
#                        [0.0,0.1,0.9]])

# print(np.argmax(predictions, axis = 1))

# prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(predictions, axis=1)])
# print(prediction_str)


