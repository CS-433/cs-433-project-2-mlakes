import spacy
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_cleaning import clean_text, remove_stop_words, remove_punctuation

# To download spacy model, run in terminal
# pip install spacy
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg


sp = spacy.load('en_core_web_sm')


def apply_preprocessing(tweets):
    """
    Applies corpus preprocessing on tweets.

    :param tweets: pd.DataFrame with the tweets.
    :return: pd.DataFrame with the original tweets and the preprocessed ones.
    """
    chunk_size = 5000
    cleaned_tweets = pd.DataFrame()
    for idx, chunk_ in tqdm(tweets.groupby(np.arange(len(tweets)) // chunk_size), desc="Data preprocessing..."):
        chunk = chunk_.copy()
        chunk = chunk.drop_duplicates(['tweet'])

        chunk['tweet'] = chunk['tweet'].apply(lambda tweet: clean_text(tweet))
        chunk['tweet'] = chunk['tweet'].apply(lambda tweet: remove_punctuation(tweet))
        chunk['tweet'] = chunk['tweet'].apply(lambda tweet: tokenize(tweet))
        chunk['tweet'] = chunk['tweet'].apply(lambda tweet: remove_stop_words(tweet))
        chunk['tweet'] = chunk['tweet'].apply(lambda tweet: lemmatization(tweet))

        chunk = chunk[chunk['tweet'].notna()]
        chunk = chunk[chunk['tweet'].str.len() > 0]
        chunk = chunk[chunk.notnull()]
        chunk = chunk.dropna()

        cleaned_tweets = pd.concat([cleaned_tweets, chunk])

    return cleaned_tweets


def tokenize(tweet):
    """
    Split text into tokens.

    :param tweet: str, the tweet
    :return: list of the tokens of the input tweet
    """
    tokens = [token for token in sp(tweet.lower())]
    return tokens


def lemmatization(tokens):
    """
    It relates all forms of a word back to its simplest form

    :param tokens: list of the tokens for a tweet
    :return: list of the lemmas of the input tweet
    """
    return [token.lemma_ for token in tokens]


def apply_preprocessing_bert(tweets, model_name='digitalepidemiologylab/covid-twitter-bert'):
    """
    Applies corpus preprocessing on tweets.

    :param tweets: pd.DataFrame with the tweets.
    :param model_name: str, the file name of the model.
    :return: pytorch Dataset with the tokens, labels and attention mask
    """
    tweets['original_tweet'] = tweets['tweet']
    tweets['tweet'] = tweets['tweet'].apply(lambda tweet: clean_text(tweet))
    tweets = tweets.drop_duplicates(['tweet'])

    # keep tweets with more than one word
    tweets = tweets[tweets['tweet'].notna()]
    tweets = tweets[tweets['tweet'].str.split().str.len() > 0]
    tweets = tweets.dropna()

    dataset = format_tensors(tweets, model_name)

    return dataset


def format_tensors(tweets, model_name):
    """
    Split text into tokens and and returns attention mask

    :param tweets: pd.DataFrame
    :param model_name: str
    :return: tuple containing the tokens and the attention mask of the input tweet
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_tuple = tweets['tweet'].apply(lambda tweet: tokenize_bert(tweet, tokenizer))

    input_ids_list = [x[0] for x in encoded_tuple]
    attention_masks_list = [x[1] for x in encoded_tuple]

    labels = torch.tensor(tweets['polarity'].values)
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)

    dataset = TensorDataset(input_ids,
                            attention_masks,
                            labels)
    return dataset


def tokenize_bert(tweet, tokenizer):
    """
    Split text into tokens and and returns attention mask

    :param tweet: str, the tweet
    :param tokenizer: model used to create the tokens
    :return:
        inputs_ids: list, tokens of the input tweet
        attention_mask: list, mask of the input tweet
    """
    encoded_dict = tokenizer.encode_plus(tweet,
                                         max_length=100,
                                         truncation=True,
                                         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                         padding='max_length',  # Pad & truncate all sentences.
                                         return_attention_mask=True,  # Construct attn. masks. Construct attn. masks.
                                         return_tensors='pt')

    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    return input_ids, attention_mask
