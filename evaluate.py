import tensorflow as tf

from data_preprocessing import (
    load_data,
    create_dataset,
    get_vocabularies
)

from model import (
    build_user_model,
    build_movie_model,
    MovieRecommendationModel
)

df = load_data(
    "../data/combined_movie_dataset.csv"
)

dataset = create_dataset(df)

user_ids, movie_titles = get_vocabularies(df)

user_model = build_user_model(user_ids)
movie_model = build_movie_model(movie_titles)

model = MovieRecommendationModel(
    user_model,
    movie_model
)

model.load_weights(
    "../saved_model/model.weights.h5"
)

test = dataset.batch(1024)

results = model.evaluate(
    test,
    return_dict=True
)

print(results)
