import pandas as pd


data = pd.read_csv("data/github_repos_dataset_20k.csv")
print(data.head())
print(data.info())
print(data.describe())
