import pandas as pd

train_label = pd.read_csv("./data/train.csv")

paths = list(set(train_label.path.to_list()))

usefull_cols = [c for c in pd.read_parquet("./data/" + paths[0]).columns if "hand" in c]

def columns_search(df, keyword):
    return df[[c for c in df.columns if keyword in c]]

def find_dominant_hand(df):
    right_hand_value = columns_search(df, "right_hand").notna().sum().sum()
    left_hand_value = columns_search(df, "left_hand").notna().sum().sum()
    return ("right_hand", right_hand_value) if right_hand_value >= left_hand_value else ("left_hand", left_hand_value)
    
def check_max_frame(path):
    df = pd.read_parquet("./data/" + path)
    max = 0
    unique_index = df.index.unique()
    for i, uniq in enumerate(unique_index):
        data = df.loc[uniq]
        if type(data) == pd.Series: data = data.to_frame().T
        dominant_hand, _ = find_dominant_hand(data)
        nb_value = columns_search(data, dominant_hand).notna().all(1).sum()
        if nb_value > max: max = nb_value
    return max

list_max_frame = []
for i, path in enumerate(paths):
    print(f"{i+1}/{len(paths)}")
    list_max_frame.append(check_max_frame(path))

print(max(list_max_frame))