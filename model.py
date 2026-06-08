import tensorflow as tf
import tensorflow_recommenders as tfrs

class MovieRecommendationModel(tfrs.Model):

    def __init__(
        self,
        user_model,
        movie_model
    ):
        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(
        self,
        features,
        training=False
    ):

        user_embeddings = self.user_model(
            features["user_id"]
        )

        movie_embeddings = self.movie_model(
            features["movie_title"]
        )

        return self.task(
            user_embeddings,
            movie_embeddings
        )

def build_user_model(user_ids):

    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=user_ids,
            mask_token=None
        ),
        tf.keras.layers.Embedding(
            len(user_ids)+1,
            64
        )
    ])

def build_movie_model(movie_titles):

    return tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=movie_titles,
            mask_token=None
        ),
        tf.keras.layers.Embedding(
            len(movie_titles)+1,
            64
        )
    ])
