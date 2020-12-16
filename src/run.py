import pandas as pd

from src.preprocessing import apply_preprocessing, apply_preprocessing_bert
from src.data_loading import load_tweets, load_test_tweets, split_data, seed_everything, split_data_bert
from src.models.bi_lstm import run_bidirectional_lstm
from src.models.machine_learning_models import run_tfidf_ml_model
from src.models.few_shot import run_zero_shot
from src.models.bert import run_bert, predict_bert

MODEL_FOLDER = '../models'
model_name = 'digitalepidemiologylab/covid-twitter-bert'
seed_everything()


def run_training(model='lg', save_model=False):
    """
    Loads data, preprocesses data and trains chosen model

    :param model: str of the model to be ran
    :param save_model: boolean indicating whether the model should be saved or not
    """
    # Load data
    tweets = load_tweets(sample=False, frac=1)

    # Data preprocessing
    print("Preprocessing...")
    if model == 'bert':
        tweets = apply_preprocessing_bert(tweets)
    else:
        tweets = apply_preprocessing(tweets)

    # Training
    if model in ['tfidf']:
        run_tfidf_ml_model(tweets=tweets[['tweet']],
                           labels=tweets[['polarity']],
                           save_model=save_model,
                           model='nb')

    elif model in ["word2vec", "glove"]:
        run_bidirectional_lstm(tweets=tweets[['tweet']],
                               labels=tweets[['polarity']],
                               save_model=save_model,
                               embeddings=model)

    elif model in ["bert"]:
        train_tweets, val_tweets = split_data_bert(tweets)
        run_bert(train_tweets=train_tweets,
                 val_tweets=val_tweets,
                 save_model=save_model)

    elif model in ["zero"]:
        train_data, val_data = split_data(tweets)
        run_zero_shot(train_tweets=train_data[['tweet']],
                      train_y=train_data[['polarity']],
                      val_tweets=val_data[['tweet']],
                      val_y=val_data[['polarity']])

    else:
        raise NotImplementedError('Please select a valid model option.')


def run_inference(final_model_name="bert_0"):
    """
    Makes predictions using the test set and the trained models.

    :param final_model_name: str of the model to make prediction with
    """
    # Load test data
    test_tweets = load_test_tweets()
    test_tweets['polarity'] = test_tweets.id  # to use same preprocessing function

    dataset = apply_preprocessing_bert(test_tweetse)

    test_ids_list, binary_preds_list = predict_bert(dataset, final_model_name)

    test_ids = np.concatenate(test_ids_list).ravel()
    binary_preds = np.concatenate(binary_preds_list).ravel()
    binary_preds = np.where(binary_preds==0, -1, binary_preds)
    results = pd.DataFrame({'Id': test_ids, 'Prediction': binary_preds})
    results.to_csv("./../predictions/predictions.csv", index=False)


def run_pipeline(pipeline='training'):
    """
    Runs either training,inference or the complete pipeline

    :param pipeline: str, `training` or `inference`
    """
    if pipeline == 'training':
        run_training(model="word2vec", save_model=False)
    elif pipeline == 'inference':
        run_inference()
    else:  # both
        run_training(model="bert", save_model=True)
        run_inference()


if __name__ == '__main__':
    run_pipeline()
