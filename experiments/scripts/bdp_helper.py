from typing import List, Tuple, Dict

import numpy as np
from scripts import encoder


def split_bdp_in_sequences(X, Y, sequence_size, overlap):
    """Splits a *single* drilling report @X, @Y into multiple sequences of length @sequence_size
    with @overlap overlapping sentences between two adjacent sequences."""

    sequence_x = list()
    sequence_y = list()

    bucket_x = list()
    bucket_y = list()

    # We ignore the overlap on the the first split as there is no previous segment
    # to recover the context from. This should be handled by the caller.
    i = 0
    global_index = 0
    while global_index < len(X):
        x, y = X[global_index], Y[global_index]
        if i == sequence_size:
            bucket_x.append(sequence_x)
            bucket_y.append(np.asarray(sequence_y))
            
            sequence_x = list()
            sequence_y = list()
            
            i = -overlap
            global_index -= overlap
        else:
            sequence_x.append(x)
            sequence_y.append(y)
            
            i += 1
            global_index += 1
    
    if len(sequence_x) > 0:
        bucket_x.append(sequence_x)
        bucket_y.append(np.asarray(sequence_y))

    return bucket_x, bucket_y


def complete_sequence(sequence_x, sequence_y, bdp_seq_size, max_sent_size):
    """Completes the sequence to have @bdp_seq_size sentences by appending zero vectors."""

    missing_rows = bdp_seq_size - sequence_x.shape[0]

    zeros = np.zeros((missing_rows, max_sent_size))
    sequence_x = np.concatenate([sequence_x, zeros], axis=0)

    zeros_labels = np.zeros(missing_rows)
    sequence_y = np.concatenate([sequence_y, zeros_labels], axis=0)
    
    return sequence_x, sequence_y


def transform_bdps_into_segments(x_bdp: List[List[str]],
                                 y_bdp: List[np.ndarray],
                                 word2index: Dict[str, int],
                                 max_sent_size: int,
                                 bdp_seq_size: int,
                                 bdp_seq_overlap: int,
                                 y_dtype = np.int32) -> Tuple[np.ndarray, np.ndarray]:
    """Divides a list of drilling reports into several segments to train a sequential model.
    :param x_bdp: A BDP, in another words, a list (bdp) of lists (sentences) of words
    :param y_bdp: A list of labels, one for each entry in the bdp
    :param word2index: Maps words to their corresponding IDs
    :param max_sent_size: The usual size of a sentence in x_bdp
    :param bdp_seq_size: The contiguous amount of entries to form a sequence for prediction
    :param bdp_seq_overlap: The amount of common entries in two consecutive sequences
    :returns A list of sequences and their labels to train a classifier
    """
    all_segments_x = list()
    all_segments_y = list()

    for i, (bdp, labels) in enumerate(zip(x_bdp, y_bdp)):
        # dividing a bdp into segments
        seq_x, seq_y = split_bdp_in_sequences(bdp,
                                              labels,
                                              sequence_size=bdp_seq_size,
                                              overlap=bdp_seq_overlap)

        # converting each sentence to a list of word IDs
        seq_x = [encoder.sentence_to_index(sentence, word2index, None, max_sent_size)
                 for sentence in seq_x]

        # some sequences (usually the last one) may not have enough samples, so we
        # pad them with trailing zeros, which are ignored by the model later on        
        padded = [complete_sequence(x, y, bdp_seq_size, max_sent_size) for x, y in zip(seq_x, seq_y)]

        # now split the padded, which is a list of lists, into two lists
        seq_x, seq_y = zip(*padded)

        all_segments_x.extend(seq_x)
        all_segments_y.extend(seq_y)

    all_segments_x = np.asarray(all_segments_x, dtype=np.int32)
    all_segments_y = np.asarray(all_segments_y, dtype=y_dtype)

    return all_segments_x, all_segments_y
