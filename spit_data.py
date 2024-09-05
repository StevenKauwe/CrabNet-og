import os

import pandas as pd

data_dir = "data/materials_data/oxides"
filename = "data_v0_preprocessed_None.csv"

df = pd.read_csv(os.path.join(data_dir, filename))
df.columns = ["formula", "target"]

df_train = df.sample(frac=0.8, random_state=0)
df_val_and_test = df.drop(df_train.index)
df_val = df_val_and_test.sample(frac=0.5, random_state=0)
df_test = df_val_and_test.drop(df_val.index)

df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(data_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
