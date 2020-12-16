import pandas as pd
import os
from flair.models.text_classification_model import TARSClassifier
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer


def predict_few_shot(tweet, classifier):
    """
    Feeds a tweet to a classifier and determines which class it belongs in
    @param tweet: the tweet to be classified
    @param classifier: the classifier
    @return: the class that the tweet belongs in
    """
    sentence = Sentence(tweet["tweet"])
    classifier.predict(sentence)
    return sentence.annotation_layers["label"][0].value


def create_sentences(tweet):
    """
    Creates a sentence object out of a tweet and brings it to the proper format for the classifier
    @param tweet: the tweet to be processed
    @return: a Sentence object with the appropriate attributes for the classifier
    """
    output = "negative" if tweet["output"] == 0 else "positive"
    return Sentence(tweet["tweet"]).add_label('positive_or_negative', output)


def run_zero_shot(train_tweets, train_y, val_tweets, val_y):
    """
    Performs the training of the zero shot learning model

    @param train_tweets: the tweets that will be used for training
    @param train_y: the training labels
    @param val_tweets: the tweets that will be used for validation
    @param val_y: the validation labels
    @return: None
    """
    # 1. Load our pre-trained TARS model for English
    print("Zero shot")
    # download https://nlp.informatik.hu-berlin.de/resources/models/tars-base/tars-base.pt
    tars = TARSClassifier.load(
        os.path.join(os.path.dirname(__file__), "..", "..","saved_models", "tars-base.pt")
    )

    train_tweets["output"] = train_y.iloc[:]
    train = train_tweets.apply(create_sentences, axis = 1).tolist()
    train = SentenceDataset(train)

    val_tweets["output"] = val_y.iloc[:]
    val = val_tweets.apply(create_sentences, axis=1).tolist()
    val = SentenceDataset(val)

    corpus = Corpus(train=train, test=val)

    tars.add_and_switch_to_new_task("POSITIVE_NEGATIVE", label_dictionary=corpus.make_label_dictionary())

    trainer = ModelTrainer(tars, corpus)

    # 4. train model
    trainer.train(base_path='../../data/zero_shot',  # path to store the model artifacts
                  learning_rate=0.02,  # use very small learning rate
                  mini_batch_size=16,  # small mini-batch size since corpus is tiny
                  max_epochs=10,  # terminate after 10 epochs
                  #train_with_dev=True,
                  )

    print("DONE TRAINING")
    tars = TARSClassifier.load('../../data/zero_shot/final-model.pt')

    val_tweets["pred"] = val_tweets.apply(predict_few_shot, args = (tars,), axis = 1)
    val_tweets["pred"] = val_tweets["pred"].apply(lambda x : 1 if x == "positive" else -1)

    pred = pd.DataFrame(list(val_tweets["pred"]), columns=['Prediction'])
    pred.index += 1
    pred.insert(0, 'Id', pred.index)

    pred.to_csv("../../data/predictions/zero_shot_pred.csv", index=False)
