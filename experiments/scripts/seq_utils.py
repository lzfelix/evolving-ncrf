from typing import List, Tuple, Dict, Optional

import sklearn
import numpy as np
from sk_seqtools import SequenceLabelEncoder

from scripts import bdp_helper
from scripts import loader
from scripts import preprocessing as pre


def load_segments(files: List[str],
                  word2index: Dict[str, int],
                  sentence_len: int,
                  segment_size: int,
                  overlap: int,
                  vocab,
                  other_classes: Optional[List[str]],
                  classmap: Optional[Dict[str, str]] = None, 
                  lencoder: SequenceLabelEncoder = None,
                  mode: str = 'ds2',
                  shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, SequenceLabelEncoder]:

    x_bdps, y_bdps = loader.load_from_some_excels(['../../data/' + f for f in files],
                                                  mode=mode,
                                                  other_classes=other_classes,
                                                  classmap=classmap)
    x_bdps = [pre.advanced_pre(bdp, vocab) for bdp in x_bdps]

    if not lencoder:
        lencoder = SequenceLabelEncoder().fit(y_bdps, pad_label='<PADL>')
    y_bdps = lencoder.transform(y_bdps)

    xa, ya = bdp_helper.transform_bdps_into_segments(x_bdp=x_bdps,
                                                     y_bdp=y_bdps,
                                                     word2index=word2index,
                                                     max_sent_size=sentence_len,
                                                     bdp_seq_size=segment_size,
                                                     bdp_seq_overlap=overlap)
    ya = ya[:, :, None]
    if shuffle:
        xa, ya = sklearn.utils.shuffle(xa, ya)
    return xa, ya, lencoder
