import pandas as pd

df = pd.read_csv("data/Final_dataset.csv", low_memory=False)

print(df.columns.tolist())