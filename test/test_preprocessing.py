import unittest
import pandas as pd
import spacy

from src.preprocessing import tokenize, lemmatization, apply_preprocessing, apply_preprocessing_bert


class TestPreprocessing(unittest.TestCase):

    def test_tokenize(self):
        text = "This is an awesome text"
        tokens = tokenize(text)

        self.assertEqual(len(tokens), 5)
        self.assertIs(type(tokens), list)
        for token in tokens:
            self.assertIs(type(token), spacy.tokens.token.Token)

    def test_lemmatization(self):
        text = "seems like awesome text"
        tokens = tokenize(text)
        lemmas = lemmatization(tokens)

        self.assertEqual(len(lemmas), len(tokens))
        self.assertIs(type(lemmas), list)
        for token in lemmas:
            self.assertIs(type(token), str)

    def test_apply_preprocessing(self):
        data = pd.DataFrame(["Dave watched as the forest burned up on the hill",
                             ":) :( :|)",
                             "The car had been hastily packed and Marta was inside trying to round up the pets",
                             "<user> and <url>",
                             "you are ..."], columns=['tweet'])

        preprocessed_data = apply_preprocessing(data)

        self.assertEqual(3, len(preprocessed_data))
        self.assertEqual(2, len(preprocessed_data.columns))

    def test_apply_preprocessed_bert(self):
        data = pd.DataFrame({'tweet': [
            "Dave watched as the forest burned up on the hill",
            ":) :( :|)",
            "The car had been hastily packed and Marta was inside trying to round up the pets",
            "<user> and <url>",
            "you are ..."],
            'polarity': [1, 0, 0, 1, 1]})
        preprocessed_data = apply_preprocessing_bert(data)

        self.assertEqual(len(preprocessed_data), 5)


if __name__ == '__main__':
    unittest.main()
