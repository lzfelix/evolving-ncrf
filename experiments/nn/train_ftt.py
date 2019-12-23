import sys
if '../' not in sys.path:
    sys.path.append('../')

import json
import argparse
import functools
from pathlib import Path

import tqdm
import fastText
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from scripts import ftt_utils
from scripts import loader
from scripts import preprocessing as pre


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Trains the fastText baseline ' +
                                     'and prepares embedding training.')
    parser.add_argument('ds_name', help='Name of the dataset to be used [dsC/dsD]',
                        type=str)
    dataset_name = parser.parse_args().ds_name

    # Getting dataset specifics
    ds_specs = loader.load_specs(dataset_name)
    MODE = ds_specs['mode']
    ALL_FILES = ds_specs['all_files']
    TRN_FILES = ds_specs['trn_files']
    DEV_FILES = ds_specs['dev_files']
    TST_FILES = ds_specs['tst_files']
    OTHER_CLASSES = ds_specs['other_classes']
    CLASS_MAP = ds_specs['class_map']    

    # Printing the dataset JSON specs for debugging
    print(json.dumps(ds_specs, indent=4))
    
    loading_fn = functools.partial(ftt_utils.load_data,
                                   mode=MODE,
                                   preprocessing=pre.keep_numbers,
                                   other_classes=OTHER_CLASSES,
                                   classmap=CLASS_MAP)

    # Figuring out 
    x_trn, y_trn = loading_fn(files=TRN_FILES)
    x_dev, y_dev = loading_fn(files=DEV_FILES)
    x_tst, y_tst = loading_fn(files=TST_FILES)
    
    lencoder = LabelEncoder()
    y_trn = lencoder.fit_transform(y_trn)
    y_dev = lencoder.transform(y_dev)
    y_tst = lencoder.transform(y_tst)

    trn_file = ftt_utils.persist_ftt(x_trn, y_trn)
    dev_file = ftt_utils.persist_ftt(x_dev, y_dev)
    tst_file = ftt_utils.persist_ftt(x_tst, y_tst)

    print(f'fastText trn: {trn_file.name}')
    print(f'fastText val: {dev_file.name}')
    print(f'fastText tst: {tst_file.name}')

    all_accs = list()
    all_uprecalls = list()
    all_mprecalls = list()

    # Train the model 15 times for statistical comparison later on
    for i in tqdm.trange(15):
        model = fastText.train_supervised(input=trn_file.name,
                                          lr=0.1,
                                          dim=50,
                                          ws=5,
                                          epoch=8,
                                          minCount=20,
                                          wordNgrams=1)

        y_tst_hat = ftt_utils.predict(model, x_tst)
        all_accs.append(accuracy_score(y_tst, y_tst_hat))

        all_uprecalls.append(ftt_utils.precall(y_tst, y_tst_hat, 'micro'))
        all_mprecalls.append(ftt_utils.precall(y_tst, y_tst_hat, 'macro'))

    ftt_utils.print_report(all_accs, all_uprecalls, all_mprecalls)

    output_vecfile = f'../../embeddings/{dataset_name}_model'
    print('To train the fastText embedding used in the downstream network, use:')
    print(f'\tfasttext supervised -input {trn_file.name} -output {output_vecfile} -lr 0.1 -minCount 20 -epoch 8 -dim 50 -wordNgrams 1 -ws 5')
    print(f'\tpython -c "from gensim.models import KeyedVectors as kv; kv.load_word2vec_format(\'{output_vecfile}.vec\').save(\'{output_vecfile}.gensim\')"')
    print('Success!')
