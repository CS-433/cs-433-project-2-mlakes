import emoji
import re
import string
import unidecode
import unicodedata
from html.parser import HTMLParser

FLAGS = re.MULTILINE | re.DOTALL


def clean_text(text):
    """
    Wrapper function that uses all the related cleaning function to clean a single tweet

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    text = text.lower()  # already lowercase
    text = standardize_text(text)
    text = remove_twitter_syntax(text)
    text = translate_emojis(text)
    text = remove_excess(text)
    text = treat_punctuation(text)
    text = decontract(text)
    text = extend_slang(text)
    text = text.strip()  # removes extra white spaces
    return text


def standardize_text(text):
    """
    Replaces \r, \n and \t with white spaces
    removes all other control characters and unicode symbols
    replaces accents and foreign letters

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    control_char_regex = re.compile(r'[\r\n\t|~]+')
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, ' ', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = unidecode.unidecode(text)  # removes accented characters
    # unicode symbols
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'So')
    text = re.sub(' +', ' ', text, flags=FLAGS)
    return text.strip()


def remove_twitter_syntax(text):
    """
    Removes special words or syntax from Twitter, including RT, users, urls

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    html_parser = HTMLParser()
    text = html_parser.unescape(text)
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text, flags=FLAGS)  # remove urls
    text = re.sub(r"@\w+", "", text, flags=FLAGS)  # remove users
    text = re.sub(r"<user>", "", text, flags=FLAGS)
    text = re.sub(r"<url>", "", text, flags=FLAGS)
    text = re.sub('RT[\s]+', "", text, flags=FLAGS)  # removes RT
    text = re.sub('rt[\s]+', "", text, flags=FLAGS)  # removes RT
    text = re.sub(' +', ' ', text, flags=FLAGS)
    return text.strip()


def translate_emojis(text):
    """
    Translates emojis characters into words

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`-]?"
    # emojis
    text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "positive", text, flags=FLAGS)
    text = re.sub(r"{}{}p+".format(eyes, nose), "positive", text, flags=FLAGS)
    text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "negative", text, flags=FLAGS)
    text = re.sub(r"{}{}[oO0]+|[oO0]+{}{}".format(eyes, nose, nose, eyes), "surprise", text, flags=FLAGS)
    text = re.sub(r"{}{}[*]+|[*]+{}{}".format(eyes, nose, nose, eyes), "kiss", text, flags=FLAGS)
    text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "neutral", text, flags=FLAGS)
    text = re.sub(r"/", " / ", text, flags=FLAGS)
    text = re.sub(r"<3", "heart", text, flags=FLAGS)

    # get words for emojis
    text = emoji.demojize(text, delimiters=(' ', ' '))
    return text.strip()


def remove_excess(text):
    """
    Removes extra letters at the end of the words and numbers

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "", text, flags=FLAGS)  # remove numbers
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 ", text, flags=FLAGS)  # removes exteeeended words
    text = re.sub(' +', ' ', text, flags=FLAGS)
    return text.strip()


def treat_punctuation(text):
    """
    Standardizes punctuation replaces & for the word and, keeps the last
    punctuation sign when there are many consecutive signs

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    # standardize punctuation
    transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
    text = ''.join([unidecode.unidecode(t) if unicodedata.category(t)[0] == 'P' else t for t in text])
    text = re.sub(r"’", "'", text, flags=FLAGS)
    text = re.sub(r"&", "and", text, flags=FLAGS)
    text = text.translate(transl_table)
    text = text.replace('…', '...')
    text = re.sub('([!?()]) ([!?()])', r'\1\2', text)  # removes spaces between punctuation
    text = re.sub(r"([\?#@+,<>%~`!$&\(\):;]){2,}", r"\1", text, flags=FLAGS)  # removes repeated punctuation
    return text


def decontract_word(phrase):
    """
    Decontracts most common contractions

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"ma'am", "madam", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase


def decontract(text):
    """
    Decontracts words in phrases

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    words = text.split()
    text = " ".join([decontract_word(word) for word in words])
    return text


def extend_slang(text):
    """
    Replaces slang with english words

    :param text: tweet to be cleaned
    :return: str with the clean tweet
    """
    contraction_dictionary = {"bro": "brother",
                              "lil": "little",
                              "dunno": "i do not know",
                              "tmr": "tomorrow",
                              "cause": "because",
                              "cos": "because",
                              "u": "you",
                              "yu": "you",
                              "ur": "you are",
                              "im": "i am",
                              "=": "equals",
                              "fck": "fuck",
                              "gonna": "going to",
                              "wanna": "want to",
                              "gotta": "got to",
                              "goodmorning": "good morning",
                              "thx": "thanks",
                              "thanx": "thanks",
                              "plz": "please",
                              "pls": "please",
                              "dont": "do not",
                              "vid": "video",
                              "gr8": "great",
                              "rt": "",
                              "isnt": "is not",
                              "imma": "i am going to",
                              "ima": "i am going to",
                              "atm": "at the moment",
                              "wbu": "what about you",
                              "btw": "by the way",
                              "brb": "i will be right back",
                              "asap": "as soon as possible"
                              }
    # fix specific words
    words = text.split()
    text = " ".join([contraction_dictionary[word] if word in contraction_dictionary else word for word in words])
    return text


def remove_stop_words(tokens):
    """
    Removes stop words according to ntlk.stop_words library.

    :param tokens: list of the tokens for a tweet
    :return: list of the cleaned tokens of the input tweet
    """
    text = [token for token in tokens if not token.is_stop]
    return text


def remove_punctuation(text):
    """
    Removes all punctuation from a sentence.

    :param text: str, the tweet
    :return: str, the tweet without punctuation
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(' +', ' ', text, flags=FLAGS)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
