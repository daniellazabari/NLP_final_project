import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from collections import Counter
from nltk.tokenize import word_tokenize

def create_vocab(data, vocab_filepath, min_count=2):
    vocab = Counter()
    data = [s.split() for s in data]
    # Add all words from all sentences to vocab
    for sentence in data:
        vocab.update(sentence)
    # Remove words that occur less than min_count times
    tokens = [k for k, c in vocab.items() if c >= min_count]
    save_vocab(tokens, vocab_filepath)
    return tokens

def save_vocab(vocab, vocab_filepath):
    vocab = '\n'.join(vocab)
    file = open(vocab_filepath, 'w')
    file.write(vocab)
    file.close()

def remove_oov_from_sentence(sentence, vocab):
    tokens = sentence.split()
    tokens_filtered = [w for w in tokens if w in vocab]
    return [tokens_filtered]

def filter_oov(data, vocab):
    data = data.to_list()
    filtered_data = []
    for sentence in data:
        sentence = remove_oov_from_sentence(sentence, vocab)
        filtered_data += sentence
    return filtered_data

def build_and_train_w2v(train_data, vocab_filepath, epochs=5, size=100, window=5, min_count=2, seed=42):
    '''
    Builds a Word2Vec model from the given data and trains it.
    '''
    vocab = create_vocab(train_data, vocab_filepath, min_count)
    print("Created vocab")
    train_data = filter_oov(train_data, vocab)
    print("Filtered data")

    model = Word2Vec(vector_size=size, window=window, min_count=min_count, seed=seed, epochs=epochs)
    model.build_vocab(train_data, progress_per=1000)

    model.train(train_data, total_examples=model.corpus_count, epochs=epochs, report_delay=1)

    return model

def save_w2v(model: Word2Vec, filename: str):
    '''
    Saves a Word2Vec model in ASCII (word2vec) format.
    '''
    model.wv.save_word2vec_format(filename, binary=False)

def load_embeddings(filename: str):
    '''
    Loads a Word2Vec model from a file.
    '''
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()

    embeddings = {}
    for line in lines:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings[word] = vector
    return embeddings

def create_sentence_vector(sentence, embeddings):
    '''
    Creates a sentence vector by averaging the word embeddings of the words in the sentence.
    '''
    vectors = [embeddings[word] for word in sentence if word in embeddings.keys()]
    if len(vectors) == 0:
        return np.zeros(100, dtype='float32')
    return np.mean(vectors, axis=0)

def create_sentence_vectors(X, original_y, vocab_filepath, embeddings):
    '''
    Creates sentence vectors for all sentences in the given data.
    '''
    X = filter_oov(X, vocab_filepath)
    vectors = []
    labels = []
    for i in range(len(X)):
        sentence = X[i]
        vector = create_sentence_vector(sentence, embeddings)
        vectors.append(vector)
        labels.append(original_y.values[i])
    return vectors, labels


