import re
import tempfile
import itertools
from typing import List, Tuple, Dict, Optional

import sklearn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from scripts import loader
from scripts import preprocessing as pre


def load_data(files: List[str],
              mode: str = 'ds1',
              preprocessing=pre.preprocess,
              other_classes: Optional[List[str]] = None,
              classmap: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    """Loads a list of files, replacing and removing undesired classes.
    
    # Arguments
        files: A list of complete filepaths to excels to be loaded.
        mode: The data layout in the file to be read.
        preprocessing: A function from the preprocessing module.
        other_classes: Classes in this list are not included in the dataset.
        classmap: Maps some class names to others. Optional.

    # Return
        A tuple of lists, the first is a list of all concatenated strinsg and the latter a
        list of all concatenated labels.
    """

    # Fixing up filepaths
    files = ['../../data/' + f for f in files]
    x_bdps, y_bdps = loader.load_from_some_excels(files,
                                                  mode,
                                                  other_classes=other_classes,
                                                  classmap=classmap)
    x_bdps = list(itertools.chain(*x_bdps))
    y_bdps = list(itertools.chain(*y_bdps))

    remove_newline = lambda x: re.sub(r'\n', '', x)
    x_bdps = preprocessing(x_bdps, None)
    x_bdps = [remove_newline(sample) for sample in x_bdps]
    x_bdps, y_bdps = sklearn.utils.shuffle(x_bdps, y_bdps)
    return x_bdps, y_bdps


def persist_ftt(x_bdps: List[str], y_bdps: List[str]):
    training_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    for sample, label in zip(x_bdps, y_bdps):
        sample = re.sub(r'\n', '', sample)
        training_file.write(f'__label__{label} {sample}\n')
    return training_file


def predict(model, x_samples: List[str]) -> List[str]:
    y_hat = model.predict(x_samples)[0]
    return [int(pred[0][len('__label__'):]) for pred in y_hat]
    

def precall(y_true, y_pred, avg='macro'):
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
    return pre, rec, f1


# gets precision, recall or F1 from the all_[um]precall
gather = lambda all_elements, index: [element[index] for element in all_elements]


def basic_stats(metrics):
    return np.average(metrics), np.std(metrics)


def format_metric(name, avg_std):
    print('{:20}\t{:4.4} ± {:4.4}'.format(name, *avg_std))


def compute_average_precall(precalls):
    avg_precision = basic_stats(gather(precalls, 0))
    avg_recall = basic_stats(gather(precalls, 1))
    avg_f1 = basic_stats(gather(precalls, 2))
    return avg_precision, avg_recall, avg_f1



def print_report(all_accs, all_uprecalls, all_mprecalls):
    format_metric('Accuracy', basic_stats(all_accs))

    uprecall = compute_average_precall(all_uprecalls)
    mprecall = compute_average_precall(all_mprecalls)
    print()
    format_metric('micro precision', uprecall[0])
    format_metric('micro recall', uprecall[1])
    format_metric('micro F1', uprecall[1])
    print()
    format_metric('macro precision', mprecall[0])
    format_metric('macro recall', mprecall[1])
    format_metric('macro F1', mprecall[1])
