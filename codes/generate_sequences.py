import pandas as pd
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import time


file_id = '1019715464'
meta_csv_path = os.path.abspath('../data')
csv_path = os.path.abspath('../data/csv_files')


################ READ THE META FILE ################
meta_csv_path = os.path.abspath('../data')
train_meta = pd.read_csv(os.path.join(meta_csv_path,'train.csv'))
sequence_ids = list(train_meta[train_meta['file_id'] == int(file_id)]['sequence_id'])
# print(len(sequence_ids))
####################################################

hands_data = pd.read_csv(os.path.join(csv_path,'{}_hand.csv').format(file_id), index_col='sequence_id')
### drop the frame number 
hands_data.drop(columns = 'frame', axis = 1, inplace=True)

# print(hands_data.index.unique())

assert len(sequence_ids) == len(hands_data.index.unique()), "Error: The number of unique sequences do not match in train.csv and hands_data csv files"
print("The number of unique sequences match in train.csv and hands_data csv files")

if os.path.exists(meta_csv_path):
    landmarks_path = os.path.join(meta_csv_path,'hand_landmarks')
    try:
        os.mkdir(landmarks_path)
    except FileExistsError:
        print(f"Directory '{landmarks_path}' already exists.")
 
else:
    print(f"Directory '{meta_csv_path}' does not exist.")

if os.path.exists(landmarks_path):
    file_id_path = os.path.join(landmarks_path,file_id)

    try:
        os.mkdir(file_id_path)
    except FileExistsError:
        print(f"Directory '{file_id_path}' already exists.")
else:
    print(f"Directory '{landmarks_path}' does not exist.")

for i in range(len(sequence_ids)):

    try:

        ### write the numpy array into  
        ## drop seqs with <100 frames
        # make a text file for the phrases
        
        array = np.array(hands_data.loc[sequence_ids[i]])
        array = np.nan_to_num(array, nan=0.0)
        print(array.shape)
        print('_'*30)
        # Save the array in the specified directory
        sequence_array_path = os.path.join(file_id_path, "{}.npy".format(str(sequence_ids[i])))
        np.save(sequence_array_path, array)



    except KeyError:
        print("The Sequence number {} exists in the train.csv file but is missing from the {}.perquet file.".format(str(sequence_ids[i]), file_id))
        continue















