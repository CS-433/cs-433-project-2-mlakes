import unittest
import spacy

from src.data_cleaning import standardize_text, remove_twitter_syntax, treat_punctuation, translate_emojis, \
    remove_excess, decontract, extend_slang, clean_text, remove_punctuation, remove_stop_words
from src.preprocessing import tokenize


class TestCleaning(unittest.TestCase):

    def test_standardize_text(self):
        text = """
        ä
            canción
        !
        Алёна
        """
        cleaned_text = standardize_text(text)
        # a cancion ! Aliona

        self.assertEqual(len(cleaned_text), 18)
        self.assertEqual('a cancion ! Aliona', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_remove_twitter_syntax(self):
        text = "RT This is an <user> awesome text @angeliki <url> https://github.com/geofot96"
        cleaned_text = remove_twitter_syntax(text)
        # This is an awesome text

        self.assertEqual(len(cleaned_text), 23)
        self.assertEqual('This is an awesome text', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_translate_emojis(self):
        text = "I am :) :D <3 =( :| ;-) "
        cleaned_text = translate_emojis(text)
        # I am positive positive heart negative neutral positive

        self.assertEqual(len(cleaned_text), 54)
        self.assertEqual('I am positive positive heart negative neutral positive', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_remove_excess(self):
        text = "heyyyy howww are uuu ? 46,3.1 "
        cleaned_text = remove_excess(text)
        # hey how are u ?

        self.assertEqual(len(cleaned_text), 15)
        self.assertEqual('hey how are u ?', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_treat_punctuation(self):
        text = "! y don’t u & me … ???!?!?!?! huh!?"
        cleaned_text = treat_punctuation(text)
        # ! y don't u and me . ! huh?

        self.assertEqual(len(cleaned_text), 29)
        self.assertEqual("! y don't u and me ... ! huh?", cleaned_text)
        self.assertIs(type(cleaned_text), str)


    def test_decontract(self):
        text = "ma'am i've but don't or won't"
        cleaned_text = decontract(text)
        # madam i have but do not or will not

        self.assertEqual(len(cleaned_text), 35)
        self.assertEqual('madam i have but do not or will not', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_extend_slang(self):
        text = "lil bro tmr plz thx"
        cleaned_text = extend_slang(text)
        # little brother tomorrow please thanks

        self.assertEqual(len(cleaned_text), 37)
        self.assertEqual('little brother tomorrow please thanks', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_clean_text(self):
        text = "=D ! ! tmr I've á deadlineeeee @EPFL <user> :O #HelpUs"
        cleaned_text = clean_text(text)
        # positive ! tomorrow i have a deadline surprise help us

        self.assertEqual(53, len(cleaned_text))
        self.assertEqual('positive ! tomorrow i have a deadline surprise helpus', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_remove_punctuation(self):
        text = "hey! ... ?! ,"
        cleaned_text = remove_punctuation(text)
        # hey

        self.assertEqual(3, len(cleaned_text))
        self.assertEqual('hey', cleaned_text)
        self.assertIs(type(cleaned_text), str)

    def test_remove_stop_words(self):
        text = "This is an awesome text"
        tokens = tokenize(text)
        clean_tokens = remove_stop_words(tokens)

        self.assertEqual(len(clean_tokens), 2)
        self.assertIs(type(clean_tokens), list)
        for token in clean_tokens:
            self.assertIs(type(token), spacy.tokens.token.Token)


if __name__ == '__main__':
    unittest.main()
