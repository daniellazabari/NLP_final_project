import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
import re
import unidecode
import contractions as contract
from symspellpy import SymSpell, Verbosity
import pkg_resources

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7) 
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

deselect_stop_words = ['no', 'not', 'nor']
stop_words = [word for word in stop_words if word not in deselect_stop_words]

# Remove accented characters from text, e.g. café
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

# Expand contractions, e.g. don't to do not
def expand_contractions(text):
    text = contract.fix(text)
    return text

# Remove duplicate words, e.g. I love love you to I love you
def remove_duplicate_words(text):
    regex = r'\b(\w+)(?:\W+\1\b)+' 
    return re.sub(regex, r'\1', text, flags=re.IGNORECASE)

# Remove urls
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Remove special characters (symbols, digits, etc.)
def remove_special_chars(text):
    return re.sub(r'[^a-zA-z\s]', '', text)

# Remove extra whitespaces
def remove_whitespaces(text):
    text = text.strip()
    return " ".join(text.split())

# Fix word lengthening, e.g. goooooood to good. This step doesn't necessarily correct a word to its correct form which brings us to the next step
def fix_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}") 
    return pattern.sub(r"\1\1", text)

def remove_newline(text):
    return ' '.join(text.splitlines())

# Correct spelling, e.g. helpjust to help just
def fix_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected_text = suggestions[0].term # get the first suggestio, otherwise return the original text if nothing
    return corrected_text

# Remove stopwords
def remove_stop_words(text):
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Lemmatize text 
def lemmatize_text(text):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    wn_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def preprocess_text(text,
                    accented_chars=True,
                    contractions=True,
                    duplicate_words=True,
                    extra_whitespace=True,
                    lemmatize=True,
                    lowercase=True,
                    newline=True,
                    url=True,
                    special_chars=True,
                    stop_words=True,
                    lengthening=True,
                    spelling=True):
    """
    Preprocesses the text by applying a series of text cleaning steps.
    """
    if accented_chars: # Remove accented characters
        text = remove_accented_chars(text)
    
    if contractions: # Expand contractions
        text = expand_contractions(text)
    
    if duplicate_words: # Remove duplicate words
        text = remove_duplicate_words(text)

    if lowercase: # Convert all chars to lowercase
        text = text.lower()
    
    if newline: # Remove newlines
        text = remove_newline(text)
    
    if url: # Remove URLs
        text = remove_urls(text)
    
    if special_chars: # Remove special characters
        text = remove_special_chars(text)
    
    if extra_whitespace: # Remove extra whitespaces
        text = remove_whitespaces(text)

    if lengthening: # Fix word lengthening
        text = fix_lengthening(text)

    if spelling: # Fix spelling
        text = fix_spelling(text)
    
    if stop_words: # Remove stop words
        text = remove_stop_words(text)
    
    if lemmatize: # Lemmatize text
        text = lemmatize_text(text)
    
    return text
