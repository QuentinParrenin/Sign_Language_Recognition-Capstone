import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import time


file_id = '1019715464'
file_path = '/home/dragon_warrior/Documents/ASL/data/{}.parquet'.format(file_id)
meta_csv_path = os.path.abspath('../data')
csv_path = os.path.abspath('../data/csv_files')


# def read_parquet_file(file_path):
#     try:
#         table = pq.read_table(file_path)
#         return table
#     except Exception as e:
#         print("Error reading Parquet file:", e)
#         return None

# def write_perquet_to_csv(parquet_table, file_id):
#     if parquet_table:
#         # Print the schema and some example data from the Parquet file
#         print("Parquet Schema:")
#         print(parquet_table.schema)
        
#         pandas_table = parquet_table.to_pandas()
#         pandas_table.to_csv('../data/csv_files/{}.csv'.format(file_id))

# write_perquet_to_csv(parquet_table,file_id)

############################ ONLY USE IT ONCE FOR THE TESTING ############################


train_meta = pd.read_csv(os.path.join(meta_csv_path,'train.csv'))

## use this piece of code in the final version to get all the file ids from the train/csv file. For now we can just hardcode the values
########################################################
# all_file_ids = [train_meta['file_id'].unique()]   ####
# print(all_file_ids)                               ####
########################################################


# Filter values from the second column based on the selected value in the first column
sequence_ids = list(train_meta[train_meta['file_id'] == int(file_id)]['sequence_id'])


################################################################################################################

data = pd.read_parquet(file_path)
print(data.index.unique())

def delete_columns_with_keyword(df, keyword):
    columns_to_delete = [col for col in df.columns if keyword in col]
    df = df.drop(columns=columns_to_delete, axis = 1)
    return df

### dropping all the coulmns with the keyword "face"
final_data = delete_columns_with_keyword(data,"face")
final_data = delete_columns_with_keyword(final_data,"pose")

## save the final dataframe in a csv file for final use of finger tracking #
final_data.to_csv(os.path.join(csv_path,'{}_hand.csv').format(file_id))
print(final_data.index.unique())
 
# make small numpy arrays that are much easier to read
# print(df_without_face.loc[1975433633])

