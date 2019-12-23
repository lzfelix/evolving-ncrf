import re
import logging
from typing import List

import unidecode


def preprocess(samples: List[str], vocab) -> List[str]:
    """Simply lowercase and remove multiple spaces."""

    clean_samples = list()
    for i, sample in enumerate(samples):
        try:
            sample = unidecode.unidecode(sample.lower().strip())

            # removes multiple spaces and content between brackets
            sample = re.sub(r'\s\s+', ' ', sample)
            sample = re.sub(r'[\(\[].*?[\)\]]', '', sample)

            clean_samples.append(sample)
        except AttributeError as ae:
            logging.error(i)

            logging.error(ae)
            logging.error(f'Something happened while reading sample #{i}: {sample}')
            clean_samples.append('')

    return clean_samples


def advanced_pre(samples, vocab):
    clean = list()
    samples = preprocess(samples, vocab)
    for text in samples:

        text = re.sub(r'[-+:;,.+=*|]', '', text) 
        text = re.sub(r'(\S)"', r'pol', text)      # replace 0" by the word 'pol'
        text = re.sub(r'\d+(\S\d+)?', '', text)    # replace numbers (3, 3.3, 3/3) with a single zero
        text = re.sub(r'\s/\s', ' ', text)         # replace slashes *after* removing numbers
        text = re.sub(r'(\d+)(.)', r'\2', text)    # splitting numbers from text
        text = re.sub(r'\s{2,}', ' ', text)

        if vocab:
            pruned_words = [word for word in text.split() if word in vocab]
            pruned_sentence = ' '.join(pruned_words)
        else:
            pruned_sentence = text
        clean.append(pruned_sentence)
    return clean


def keep_numbers(samples, vocab):
    clean = list()
    samples = preprocess(samples, vocab)
    for text in samples:

        text = re.sub(r'[-+:;,.+=*|]', '', text) 
        text = re.sub(r'\s{2,}', ' ', text)

        if vocab:
            pruned_words = [word for word in text.split() if word in vocab]
            pruned_sentence = ' '.join(pruned_words)
        else:
            pruned_sentence = text
        clean.append(pruned_sentence)
    return clean


def advanced_pre_no_numbers(samples, vocab):
    clean = list()
    samples = preprocess(samples, vocab)
    for text in samples:

        text = re.sub(r'[-+:;,.+=*|]', '', text) 
        text = re.sub(r'(\S)"', r'pol', text)     # replace 0" by the word 'pol'
        text = re.sub(r'\d+(\S\d+)?', '', text)   # replace numbers (3, 3.3, 3/3) with a single zero
        text = re.sub(r'\s/\s', ' ', text)        # replace slashes *after* removing numbers
        text = re.sub(r'(\d+)(.)', r'\2', text)   # splitting numbers from text
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\d', ' ', text)

        if vocab:
            pruned_words = [word for word in text.split() if word in vocab]
            pruned_sentence = ' '.join(pruned_words)
        else:
            pruned_sentence = text
        clean.append(pruned_sentence)
    return clean
