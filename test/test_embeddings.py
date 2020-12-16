import unittest
from src.embeddings import text_to_glove
from src.preprocessing import apply_preprocessing
import pandas as pd
import numpy as np


class TestEmbeddings(unittest.TestCase):

    def test_text_to_glove(self):
        embeddings_dict = {
            "sentence": np.array([1, 2, 3, 4, 5]),
            "good": np.array([1, 0, 0, 3, 2])
        }
        embedding_length = 5

        data = pd.DataFrame(["good 1 sentence",
                             "sentence",
                             "sentence",
                             "a very good"], columns=['tweet'])

        data = apply_preprocessing(data)
        max_length = data['tweet'].apply(lambda x: len(x)).max()

        vectorized_data = text_to_glove(data=data,
                                        embeddings=embeddings_dict,
                                        embedding_length=embedding_length,
                                        sequence_length=max_length)

        self.assertEqual(vectorized_data.shape, (3, 2, 5)) #Drops duplicates
        self.assertIs(type(vectorized_data), np.ndarray)


if __name__ == '__main__':
    unittest.main()
