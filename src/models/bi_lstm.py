import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding, Bidirectional
import pickle
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os

from sklearn.model_selection import KFold
from src.evaluate import plot_graphs
from src.embeddings import texts_to_sequences, text_to_glove
from src.data_loading import seed_everything


DATA_FOLDER = "./../data/"
MODEL_FOLDER = './../models/'
EMBEDDINGS_FOLDER = DATA_FOLDER + 'embeddings/'
#MODEL_FOLDER = '../models/'
#EMBEDDINGS_FOLDER = '../data/embeddings/'
seed_everything()


def create_word2vec(tweets):
    """
    It vectorizes the text of each tweet.

    :param tweets: pd.DataFrame, with the train data
    :return:
        train_v: np.array, with the training vectorized input
        train_v: np.array, with the validation vectorized input
        vocab: dict, the vocabulary used
    """
    print("Creating word2vec features...")
    file_path = DATA_FOLDER + 'vocab.pkl'
    with open(file_path, 'rb') as f:
        vocab = pickle.load(f)

    train_v = texts_to_sequences(tweets=tweets['tweet'],
                                 vocab_size=len(vocab))

    return train_v, vocab


def create_glove(tweets, max_sequence_length, embedding_length=100):
    """
    It vectorizes the text of each tweet by concatenating and padding
    the respective glove pre-trained embeddings.

    :param tweets: pd.DataFrame, with the train data
    :param max_sequence_length: int, the length of the sequence
    :param embedding_length: int, the length of the embeddings
    :return:
        train_v: np.array, with the training vectorized input
        train_v: np.array, with the validation vectorized input
        len(embeddings_dict): int, the vocabulary size
    """
    embeddings_dict = {}
    with open(os.path.join(EMBEDDINGS_FOLDER, "glove.twitter.27B.100d.txt"), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    train_v = text_to_glove(data=tweets,
                            embeddings=embeddings_dict,
                            embedding_length=embedding_length,
                            sequence_length=max_sequence_length)

    return train_v, len(embeddings_dict)


def run_bidirectional_lstm(tweets, labels, embeddings='glove', cross_validation=False, save_model=False, verbose=False):
    """
    Trains a bi-directional lstm

    :param tweets: pd.DataFrame with the features
    :param labels: pd.DataFrame with the labels
    :param embeddings: str, the type of embeddings
    :param save_model: bool
    :param cross_validation: bool
    :param verbose: bool
    """
    # Text transformation
    if embeddings == 'word2vec':
        print("\n" + "-" * 100)
        print("MODEL TO RUN: Bi-LSTM with word2vec embeddings")
        print("-" * 100 + "\n")

        tweet_vectors, vocab = create_word2vec(tweets=tweets)

    elif embeddings == 'glove':
        print("\n" + "-" * 100)
        print("MODEL TO RUN: Bi-LSTM with GloVe embeddings")
        print("-" * 100 + "\n")

        embedding_dim = 100
        sequence_length = tweets['tweet'].apply(lambda x: len(x)).max()

        tweet_vectors, vocab_size = create_glove(tweets=tweets,
                                                 embedding_length=embedding_dim,
                                                 max_sequence_length=sequence_length)
    else:
        raise ValueError('Please select a valid embedding value')

    # Modeling
    kf = KFold(n_splits=10)
    labels_ = labels.to_numpy()
    for i, (train_index, validation_index) in enumerate(kf.split(tweet_vectors)):
        print('Running cross validation on {}/{} fold.'.format(i + 1, kf.get_n_splits()))
        train_x = tweet_vectors[train_index]
        train_y = labels_[train_index]
        val_x = tweet_vectors[validation_index]
        val_y = labels_[validation_index]

        print('Train data shape: {}'.format(train_x.shape))
        print('Validation data shape: {}'.format(val_x.shape))

        if embeddings == 'word2vec':
            embedding_dim = 64

            model = Sequential()
            model.add(Embedding(len(vocab), embedding_dim))
            model.add(Dropout(0.5))
            model.add(Bidirectional(LSTM(embedding_dim)))
            model.add(Dense(1, activation='sigmoid'))
            model.summary()

        elif embeddings == 'glove':
            model = Sequential()
            model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2),
                                    input_shape=(sequence_length, embedding_dim)))
            model.add(Bidirectional(LSTM(32)))  # last layer only returns the last input
            model.add(Dense(1, activation='sigmoid'))
            model.summary()

        else:
            raise ValueError('Please select a valid embedding value')

        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6),
            # TODO maybe do some hyperparameter optimization?,
            metrics=['accuracy'])

        filepath = MODEL_FOLDER + "best_model.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='max')
        callbacks_list = [checkpoint]
        num_epochs = 100
        print("Fitting model...")
        history = model.fit(train_x, train_y,
                            epochs=num_epochs,
                            validation_data=(val_x, val_y),
                            callbacks=callbacks_list,
                            verbose=2)

        if verbose:
            plot_graphs(history, "accuracy")
            plot_graphs(history, "loss")

        pred = model.predict(val_x)
        pred = np.array([1 if p >= 0.5 else 0 for p in pred])

        print('Accuracy: {:.3f}'.format(accuracy_score(y_true=val_y, y_pred=pred)))
        print(classification_report(y_true=val_y, y_pred=pred))

        if not cross_validation:
            break

    # save model
    if save_model:
        print("Saving model...")
        model.save(MODEL_FOLDER)
        print("Done!")
