import pandas as pd

x_train_aug = pd.read_parquet('data/processed/x_train_aug.parquet')
x_val = pd.read_parquet('data/processed/x_val.parquet')
y_train_aug = pd.read_parquet('data/processed/y_train_aug.parquet')['label']
y_val = pd.read_parquet('data/processed/y_val.parquet')['label']