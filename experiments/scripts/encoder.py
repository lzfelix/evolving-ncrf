import numpy as np
from keras.preprocessing import sequence


def sentence_to_index(X, word2index, pre=None, max_sequence_len=None):
    """@from[Minuet/custom] Converts sentences as lists of words to list of word ids.
    :param X: List of sentences (space-separated words).
    :param word2index: Dict mapping each word to its ID.
    :param pre: Preprocessing functions to be applied to each word of X.
    :param max_sequence_len: Smaller sequences will be prepended with zeros and
    larger ones will be trimmed at the end. If None, no adjustment is performed.
    :returns Depends on max_sequence_len. If None, returns a list of lists
    otherwise a m-by-d matrix of word IDs (from word2index) where m is the
    amount of samples and d the word-vectors size.
    """
    
    pre = pre or (lambda x: x)
    
    all_sentences = list()
    for sentence in X:
        sentence_indices = list()

        for word in sentence.split():
            word = pre(word)
            index = word2index.get(word, word2index['<unk>'])
            sentence_indices.append(index)
            
        all_sentences.append(sentence_indices)
        
    if max_sequence_len:
        return sequence.pad_sequences(
            all_sentences,
            max_sequence_len,
            truncating='post',
            padding='post')
    else:
        return all_sentences


def encode_bdps_operations(encoder, Y):
    """Given a list of bdps labels @Y, encodes them into numbers preserving
    the BDP structure using the <encoder>."""
    return [encoder.transform(y) for y in Y]


def compute_avg_sentence_vector(sentence_ids, E):
    """Computes the sentence vector as the average of its word vectors."""

    word_vectors = np.asarray([E[wid, :]
                               for wid
                               in sentence_ids if wid > 0
                              ])

    # if there are no word vectors in the sentence, create a single blank vector
    # so the function outputs a blank vector as well
    if word_vectors.size == 0:
        word_vectors = np.zeros((1, E.shape[1]))
    amount_vectors = word_vectors.shape[1]

    return np.sum(word_vectors, axis=0) / amount_vectors


def compute_avg_sentence_vector_all_segments(segments, E):
    """Computes the average sentence vectors for all sentences in all segments."""
    segment_vectors = list()
    for i, segment in enumerate(segments):
        sentence_vectors = [
            compute_avg_sentence_vector(sentence, E)
            for sentence
            in segment
        ]

        segment_vectors.append(sentence_vectors)
    return np.asarray(segment_vectors)
