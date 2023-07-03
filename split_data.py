import pandas as pd
from os import walk
from tqdm import tqdm

from multiprocessing import Pool

def save_data(args):
    seq, df = args
    if type(df) == pd.Series: df = df.to_frame().T
    df.to_parquet(f"./data/splited_data/{seq}.parquet")

def main():
    folder = "./data/"

    for _,_,file_name in walk(folder + "train_landmarks"):
        files = file_name
        
    with Pool(16) as p:
        for i, f in enumerate(tqdm(files)):
            #print(f"\nFILES: {i}/{len(files)}")
            
            df = pd.read_parquet(folder + "train_landmarks/" + f)
            sequence_ids = df.index.unique().to_list()
            
            df_sequence_ids = [(seq, df.loc[seq]) for seq in sequence_ids]
            
            p.map(save_data, df_sequence_ids)
            
if __name__ == "__main__":
    main()