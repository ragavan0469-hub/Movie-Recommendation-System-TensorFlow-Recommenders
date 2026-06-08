import pandas as pd
import tensorflow as tf

def load_data(file_path):
    df = pd.read_csv(file_path)

    df["userId"] = df["userId"].astype(str)
    df["title"] = df["title"].astype(str)

    return df

def create_dataset(df):

    dataset = tf.data.Dataset.from_tensor_slices({
        "user_id": df["userId"].values,
        "movie_title": df["title"].values
    })

    return dataset

def get_vocabularies(df):

    user_ids = df["userId"].unique()
    movie_titles = df["title"].unique()

    return user_ids, movie_titles
