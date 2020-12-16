#!/usr/bin/env python3
import pickle

EMBEDDINGS_DIR = './../data/embeddings/'
def main():
    vocab = dict()
    file_dir = EMBEDDINGS_DIR + 'vocab_cut.txt'
    with open(file_dir) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

	file_dir =  EMBEDDINGS_DIR + 'vocab.pkl'
    with open(file_dir, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
