from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_tf_idf(tweets, n_gram):
    """
    Creates tf-idf features.
    It fits a vectorizer in the training data and applies it in validation ones.

    :param tweets: pd.Series of the tweets
    :param n_gram: int, the max number of n-gram features
    :return:
        train: pd.DataFrame with tf-idf features of the training set
        dev: pd.DataFrame with tf-idf features of the dev set
    """
    tweets_train = tweets.tolist()
    tweets_train_l = [' '.join(tweet) for tweet in tweets_train]  # convert list of tokens into str

    # Fit-transform to train set and transform to dev set
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(tweets_train_l)
    features_df = pd.DataFrame(features.todense(), columns=vectorizer.get_feature_names())

    return features_df


def texts_to_sequences(tweets, vocab_size):
    """
    Created dense vectors for the input texts

    @param tweets: The training set
    @param vocab_size: The size of the vocabulary of the training set
    @return: The preprocessed train and validation sets and labels
    """
    max_length = tweets.apply(lambda x: len(x)).max()
    trunc_type = 'post'
    padding_type = 'post'

    # fit to train data
    oov_tok = '<OOV>'  # OOV = Out of Vocabulary
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(tweets)

    # transform to train & val data
    train_sequences = tokenizer.texts_to_sequences(tweets)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return train_padded


def text_to_glove(data, embeddings, sequence_length, embedding_length=100):
    """
    It maps the glove embedding for each token in each tweet

    :param data: pd.DataFrame, with the tokenized tweets
    :param embeddings: dict, with the glove embeddings for each word
    :param sequence_length: int, the sentence max length
    :param embedding_length: int, the vector length of the word embedding
    :return: np.ndarray, with the input data with dimensions: ( , sequence_length, embedding_length)
    """

    sequences = []
    for index, row in data.iterrows():  # tweet
        glove_seq = []
        for token in row['tweet']:  # word
            # get glove only if it exists
            if embeddings.get(token, None) is not None:
                embedding = list(embeddings[token])
                glove_seq.append(embedding)

        # padding
        for zero in range(sequence_length - len(glove_seq)):
            zeros = [0] * embedding_length
            glove_seq.append(zeros)

        sequences.append(glove_seq)

    return np.array(sequences)
