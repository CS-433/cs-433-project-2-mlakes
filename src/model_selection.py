import pandas as pd

from src.data_loading import load_tweets
from src.models.machine_learning_models import run_tfidf_ml_model
from src.preprocessing import apply_preprocessing


def run_model_selection_for_ml_model():
    # Load data
    tweets = load_tweets(sample=True, frac=0.01)
    tweets = apply_preprocessing(tweets)

    train_tweets = tweets[['tweet']]
    labels = tweets[['polarity']]

    models = ['lg', 'svm', 'nb', 'rf']
    scores = list()
    for model in models:
        scores.append(run_tfidf_ml_model(tweets=train_tweets, labels=labels, model=model))

    print(pd.DataFrame(scores))


def run_model_selection_for_deep_learning_model():
    pass


if __name__ == '__main__':
    run_model_selection_for_ml_model()
