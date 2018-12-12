import pandas as pd


def read_tabular_dataset(path):
    df = pd.read_csv(path)
    tweets_a = df['tweets_a'].tolist()
    tweets_b = df['tweets_a'].tolist()
    labels = df['labels'].tolist()
    return [tweets_a, tweets_b, labels]
