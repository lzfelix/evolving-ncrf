"""Scripts to load the drilling reports"""

import sys
import glob
import json
import logging
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import tqdm
from gensim.models import KeyedVectors
from nltk.text import FreqDist


COL_X = 'text'
COL_Y = 'labels'

datadict = {
    'ds1': {
        COL_X: 'COMENTARIOS',
        COL_Y: 'OPERACAO',
    },
    'ds3': {
        COL_X: 'DESCRICAO',
        COL_Y: 'TAREFA_EAP_SAP_SINONIMO',
    },
}


class InvalidExcelError(RuntimeError):
    """Represents an error that happened while handling the Excel file"""
    pass


def validate_columns(required_columns: List[str],
                     all_columns: List[str]) -> None:
    """Check if all required_columns are present in all_columns"""

    missing_columns = [column for column in required_columns if column not in all_columns]
    if any(missing_columns):
        missing = ', '.join(missing_columns)
        raise InvalidExcelError(f'Coluna/s "{missing}" ausente/s na planilha')


def load_full_excel(filepath: str, mode: str) -> Tuple[List[str], List[str]]:
    """Given a drilling report, loads the X (text) and Y (labels) columns."""

    logging.info(f'Loading excel file {filepath}')
    data = pd.read_excel(filepath, ignore_index=True)
    
    logging.info('Excel file loaded successfully')
    
    fields = datadict.get(mode)
    if not fields:
        raise RuntimeError('Got {}, but valid file modes are: "{}"'.format(
                           mode, ' '.join(datadict.keys())))

    col_text = datadict[mode][COL_X]
    col_label = datadict[mode][COL_Y]
    validate_columns([col_text, col_label], data.columns)

    logging.info('Filtering relevant rows')

    # removing invalid columns
    data = data[data[col_text] != '']

    X = data[col_text].fillna('').tolist()
    y = data[col_label].tolist()
    return X, y


def load_data_from_excel(filepath: str,
                         mode: str) -> Tuple[list, List[str], List[str]]:
    descricoes, operacoes = load_full_excel(filepath, mode)
    indices = [None] * len(descricoes)
    return indices, descricoes, operacoes


def load_all_excel(folder_path: str,
                   merge: bool = False,
                   mode: str = 'ds1') -> Tuple[List[str], List[str]]:
    """Loads data from all excel files stored in folder @folder_path. If @merge, then a single
    list of description is returned, otherwise each BDP is a node in the returned list."""
    all_indices = list()
    all_x = list()
    all_y = list()
    
    files = glob.glob(folder_path)

    for index in tqdm.trange(len(files)):
        index, x, y = load_data_from_excel(files[index], mode)
        
        if merge:
            all_indices.extend(index)
            all_x.extend(x)
            all_y.extend(y)
        else:
            all_indices.append(index)
            all_x.append(x)
            all_y.append(y)
        
    return all_x, all_y


def load_from_some_excels(excels: List[str],
                          mode: str = 'ds1',
                          other_classes: List[str] = None,
                          classmap: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    all_x = list()
    all_y = list()

    for index in tqdm.trange(len(excels)):
        filename = excels[index]
        _, x, y = load_data_from_excel(filename, mode)
        
        # removing samples from the other classes
        if other_classes:
            R = list(filter(lambda bdp_line: (bdp_line[1] not in other_classes), zip(x, y)))
            if len(R) == 0:
                print(f'[WARN] File {excels[index]} contains only skipped entries')
                sys.stdout.flush()
                continue

            x, y = zip(*R)

        if len(x) != len(y):
            raise RuntimeError(f'Amount of entries for X is different from y for {filename}')
        elif len(x) == 0:
            print(f'[WARN] All entries skipped for file {filename}')
            continue

        if classmap:
            y = [classmap.get(label, label) for label in y]
        
        all_x.append(x)
        all_y.append(y)

    return all_x, all_y


def load_embeddings_matrix(embeddings: KeyedVectors, X: List[str], top_k: int = 0):
    """Given a gensim word embedding object, outputs the embedding matrix E, the word2index and missing words.
    :param embeddings
    :param X: List of sentences
    :param top_k: Keeps only the top_k most frequent word embeddings.
    :param normalize_unk
    :returns: E, w2i, missing, where the embedding matrix E has the i-th row corresponding to the word from
    w2i[some_word], missing number of words without embedding.
    """

    word2index = dict()
    word2index['<pad>'] = 0
    word2index['<unk>'] = 1

    embeddings_table = list()

    all_words = itertools.chain(*[sentence.split() for sentence in X])
    all_words = list(all_words)
    vocab = FreqDist(all_words)

    # keeping just topk words
    if top_k > 0:
        # each entry is a pair (word, freq), we just need the words here
        print(f'Keeping top {top_k} words.')
        allowed_words = [entry[0] for entry in vocab.most_common(top_k)]
    else:
        allowed_words = [entry[0] for entry in vocab.items()]

    # adding PAD and UNK word vectors
    embeddings_table.append(np.zeros(embeddings.vector_size))

    mean_vector = embeddings.vectors.mean(axis=0)
    embeddings_table.append(mean_vector)

    # retrieving from the embedding matrix
    missing = set()
    for word in allowed_words:
        try:
            embeddings_table.append(embeddings[word])
            word2index[word] = len(word2index)
        except KeyError:
            missing.add(word)

    embeddings_table = np.asarray(embeddings_table)
    return embeddings_table, word2index, missing


def load_specs(dataset_name: str) -> dict:
    file_specs = json.load(Path('../../data/datasets.json').open('r'))
    return file_specs[dataset_name]
