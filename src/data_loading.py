import pandas as pd
import os
import random
import numpy as np
import torch
from torch.utils.data import random_split

DATA_FOLDER = '../data'


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def load_tweets(sample=True, frac=1):
    """
    Reads train data from both classes and returns their concatenated dataset.

    :param sample: boolean, `True` if a sample size of data to be loaded, `False` otherwise
    :param frac: float, fraction of data to be returned
    :return: pd.DataFrame with the training data (features and labels)
    """
    positive_file = 'train_pos.txt' if sample else 'train_pos_full.txt'
    negative_file = 'train_neg.txt' if sample else 'train_neg_full.txt'

    positive_tweets = pd.read_csv(os.path.join(DATA_FOLDER, positive_file),
                                  delimiter="\t", header=None, names=['tweet'])
    positive_tweets['polarity'] = 1

    negative_tweets = pd.read_csv(os.path.join(DATA_FOLDER, negative_file),
                                  delimiter="\t", header=None, names=['tweet'])
    negative_tweets['polarity'] = 0

    tweets = pd.concat([positive_tweets, negative_tweets]).reset_index(drop=True)

    print('Positive tweets: {:.0f}'.format(len(positive_tweets) * frac))
    print('Negative tweets: {:.0f}'.format(len(negative_tweets) * frac))
    print('Most frequent label model: {:.3f}'.format(max(len(positive_tweets), len(negative_tweets)) / len(tweets)))

    return tweets.sample(frac=frac, random_state=1).reset_index(drop=True)


def load_test_tweets():
    """
    Loads the test dataset.

    :return: pd.DataFrame with the test data (features and ids)
    """
    test_tweets = pd.read_csv(os.path.join(DATA_FOLDER, 'test_data.txt'),
                              delimiter="\t", header=None, names=['tweet'])

    test_tweets = test_tweets['tweet'].str.split(',', n=1, expand=True).rename(columns={0: 'id', 1: 'tweet'})

    return test_tweets


def split_data_bert(dataset, ratio=0.95):
    """
    Splits the data into training and validation datasets based on a given ratio.

    :param tweets: pd.DataFrame with the training data (features and labels)
    :param ratio: float, the splitting ratio
    :return:
        train_tweets: pd.DataFrame with the training features
        val_tweets: pd.DataFrame with the validation features
    """
    # Calculate the number of samples to include in each set.
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_tweets, val_tweets = random_split(dataset, [train_size, val_size])

    return train_tweets, val_tweets


def split_data(data, ratio=0.9):
    """
    Splits the data into training and validation datasets based on a given ratio.

    :param data: pd.DataFrame with the training data (features and labels)
    :param ratio: float, the splitting ratio
    :return:
        train_data: pd.DataFrame with the training data (features and labels)
        val_data: pd.DataFrame with the validation data (features and labels)
    """
    train_size = round(ratio * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:]

    return train_data, val_data


def load_preprocessed_tweets():
    """  Loads the already preprocessed datasets from disc.  """
    train_x = pd.read_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'train_x.csv'), delimiter=",", header=None)
    train_y = pd.read_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'train_y.csv'), delimiter=",", header=None)
    val_x = pd.read_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'val_x.csv'), delimiter=",", header=None)
    val_y = pd.read_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'val_y.csv'), delimiter=",", header=None)

    return train_x, train_y, val_x, val_y


def store_preprocessed_tweets(train_x, val_x, train_y, val_y):
    """  Stores the already preprocessed datasets to disc.  """
    train_x.to_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'train_x.csv'), sep=',')
    val_x.to_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'val_x.csv'), sep=',')
    train_y.to_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'train_y.csv'), sep=',')
    val_y.to_csv(os.path.join(DATA_FOLDER, 'preprocessed_data', 'val_y.csv'), sep=',')
