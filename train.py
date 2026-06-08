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

dataset = dataset.shuffle(
    len(df)
)

train = dataset.take(
    int(len(df)*0.8)
).batch(1024)

test = dataset.skip(
    int(len(df)*0.8)
).batch(1024)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        0.001
    )
)

model.fit(
    train,
    validation_data=test,
    epochs=5
)

model.save_weights(
    "../saved_model/model.weights.h5"
)
