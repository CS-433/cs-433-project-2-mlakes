import pandas as pd
import itertools as it

from src.data_loading import load_tweets
from src.models.machine_learning_models import run_tfidf_ml_model
from src.models.bi_lstm import run_bidirectional_lstm
from src.preprocessing import apply_preprocessing


def settings_combinations(search_space):
    """
    It creates all the combinations of values between the elements of input lists.
    e.g.
        Input:
            {'paramA': ['value1', 'value2'],
             'paramB': ['value3', 'value4']}
        Output:
            [['value1', 'value3'],
            ['value1', 'value4'],
            ['value2', 'value3'],
            ['value2', 'value4']]

    :param search_space: dict of lists, with the possible values for each hyper-parameter.
    :return:
        settings: list of lists with the different combinations of settings.
    """
    params = sorted(search_space)
    settings = list(it.product(*(search_space[param] for param in params)))
    return settings


def run_model_selection_for_ml_model():
    """  It runs model selection for the tf-idf models to decide the best classifier.  """

    # Load data
    tweets = load_tweets(sample=True, frac=1)
    tweets = apply_preprocessing(tweets)

    train_tweets = tweets[['tweet']]
    labels = tweets[['polarity']]

    models = ['lg', 'svm', 'nb', 'rf']
    scores = list()
    for model in models:
        scores.append(run_tfidf_ml_model(tweets=train_tweets, labels=labels, model=model))

    print(pd.DataFrame(scores))


def run_model_selection_for_deep_learning_model():
    """  It runs model selection for the bi-lstm models to decide the best hyper-parameters.  """

    # Hyper-parameters to test
    model_parameters = {
        'batch_size':  [32, 64, 128],
        'learning_rate': [0.001, 0.0001, 0.00001, 0.000001],
        'embeddings': ['glove', 'word2vec']
    }

    # Calculate the combinations of hyper-parameters values to produce the settings that model selection will run
    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings' .format(len(model_settings)))

    # Load data
    tweets = load_tweets(sample=True, frac=1)
    tweets = apply_preprocessing(tweets)

    scores = list()
    for setting in model_settings:
        bs = setting[0]
        e = setting[1]
        lr = setting[2]
        scores.append(run_bidirectional_lstm(tweets=tweets[['tweet']],
                                             labels=tweets[['polarity']],
                                             batch_size=bs,
                                             lr=lr,
                                             save_model=False,
                                             embeddings=e))

    print(pd.DataFrame(scores))


if __name__ == '__main__':
    run_model_selection_for_ml_model()
    run_model_selection_for_deep_learning_model()
