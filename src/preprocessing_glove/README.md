# Project Text Sentiment Classification

Submission system environment setup:

1. The dataset is available from the AIcrowd page, as linked in the PDF project description

 Download the provided datasets `twitter-datasets.zip`.

2. Generating Word Embeddings: 

Load the training tweets given in `pos_train.txt`, `neg_train.txt` (or a suitable subset depending on RAM requirements), and construct a a vocabulary list of words appearing at least 5 times. This is done running the following commands. Note that the provided `cooc.py` script can take a few minutes to run, and displays the number of tweets processed.

```bash
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```




