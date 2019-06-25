from typing import List, Dict, Tuple, Optional

import sklearn
import numpy as np
from sk_seqtools import SequenceLabelEncoder

from scripts import loader
from scripts import encoder
from scripts import bdp_helper
from scripts import preprocessing as pre


def load_split(files: List[str],
               word2index: Dict[str, int],
               E: np.ndarray,
               sentence_len: int,
               segment_size: int,
               overlap: int,
               vocab,
               preprocessing = pre.preprocess,
               other_classes: Optional[List[str]] = None,
               lencoder: SequenceLabelEncoder = None) -> Tuple[np.ndarray, np.ndarray, SequenceLabelEncoder]:
    """
    Loads a dataset split and makes segments out of it. Samples are automatically shuffled.
    :param files: List of filepaths
    :param word2index: ...
    :para E: The word embedding matrix where E[word2index[w]] is a row with w's embedding
    :param sentence_len: How many words at most each sentence should contain
    :param segment_size: The size of a segment (sequence) for classification
    :param overlap: Overlap of samples between two adjacent segments
    :param other_classes: Samples from these classes will be disregarded
    :param lencoder: The label encoder to transform the samples. If not provided, a new
    one is created
    :return [a, b, c], where a is a tensor of samples with shape [n_sequences, segment_size, embedding_size],
    b has shape [n_sequences, segment_size, 1] and c is the label encoder created (if not supplied) or the
    same encoder provided to the function.
    """
    x_bdps, y_bdps = loader.load_from_some_excels(['../' + f for f in files],
                                                  mode='ds1',
                                                  other_classes=other_classes)
    x_bdps = [preprocessing(bdp, vocab) for bdp in x_bdps]

    if not lencoder:
        print('Creating new label encoder')
        lencoder = SequenceLabelEncoder().fit(y_bdps, pad_label='<PADL>')
    ya = lencoder.transform(y_bdps)

    xa, ya = bdp_helper.transform_bdps_into_segments(x_bdp=x_bdps,
                                                     y_bdp=ya,
                                                     word2index=word2index,
                                                     max_sent_size=sentence_len,
                                                     bdp_seq_size=segment_size,
                                                     bdp_seq_overlap=overlap)

    # computing sentence vectors as average
    xa = encoder.compute_avg_sentence_vector_all_segments(xa, E)
    
    # Adding the third axis to the labels
    ya = ya[:, :, None]
    
    # Shuffling samples
    xa, ya = sklearn.utils.shuffle(xa, ya)
    return xa, ya, lencoder
