import sys
if '../scripts/' not in sys.path:
    sys.path.append('../')

import os
import argparse
import functools
from typing import *
from datetime import datetime

import sklearn
import numpy as np
from sk_seqtools import SequenceLabelEncoder
from gensim.models import KeyedVectors

from keras import models
from keras import layers
from keras import callbacks
from keras import regularizers
from keras import optimizers as optim
from scripts.layers import AttentionLayer
from scripts.lr_scheduler import SGDRScheduler

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from scripts import loader
from scripts import metrics
from scripts import seq_utils
from scripts import bdp_helper
from scripts import preprocessing as pre


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('ds', help='Name of the dataset [dsC/dsD]', type=str)
    parser.add_argument('embeddings', help='Path to the word embeddings', type=str)
    parser.add_argument('destination', help='Folder where the trained binaries are going to be saved', type=str)
    parser.add_argument('--use_crf', help='Uses CRF instead of Softmax in the output layer', action='store_true')
    parser.add_argument('--use_context', help='Uses second level LSTM to exchange context info', action='store_true')
    parser.add_argument('-n_models', help='Train the model n times for statistical comparison', type=int, default=15)
    args = parser.parse_args()

    print(f'Dataset:     {args.ds}')
    print(f'Embedidngs:  {args.embeddings}')
    print(f'# Models:    {args.n_models}')
    print(f'Use Context: {args.use_context}')
    print(f'Use CRF:     {args.use_crf}')
    
    return args


def build_model(sentence_len,
                segment_len,
                hidden_size,
                dropout_prob,
                n_classes,
                crf_lambda,
                use_crf=False,
                use_context=True):
    words_in = layers.Input(
        shape=(segment_len, sentence_len),
        name='words-in'
    )

    # embedding
    embedded = layers.Embedding(
        input_dim=E.shape[0],
        output_dim=E.shape[1],
        weights=[E], name='embedding',
        trainable=False,
        mask_zero=True
    )(words_in)

    masked_sentence = layers.Masking(
        name='episode-mask'
    )(embedded)
    
    masked_sentence = layers.Dropout(rate=0.2, noise_shape=(None, BDP_SEQ_SIZE, MAX_SENT_LEN, 1), name='embedding-drop')(masked_sentence)
    # > (?, segment, sentence, embedding)

    # encoding - drops inputs
    birnn = layers.Bidirectional(layers.GRU(
        hidden_size,
        dropout=dropout_prob,
        recurrent_dropout=dropout_prob,
        return_sequences=True
    ))

    time_rnn = layers.TimeDistributed(
        birnn,
        name='time-rnn'
    )(masked_sentence)
    # > (?, segment, sentence, embedding)
    
    time_rnn = layers.TimeDistributed(AttentionLayer(), name='attention')(time_rnn)
    time_rnn = layers.BatchNormalization(name='norm')(time_rnn)
    
    if use_context:
        time_rnn = layers.Bidirectional(layers.GRU(hidden_size, dropout=dropout_prob, recurrent_dropout= dropout_prob, return_sequences=True))(time_rnn)
    time_rnn = layers.Dropout(dropout_prob)(time_rnn)

    # output
    if use_crf:
        crf = CRF(
            n_classes,
            sparse_target=True,
            name='crf',
            kernel_regularizer=regularizers.l2(crf_lambda),
            chain_regularizer=regularizers.l2(crf_lambda)
        )
        output = crf(time_rnn)
        loss = crf.loss_function
        accuracy = [crf.accuracy]
    else:
        output = layers.Dense(n_classes, activation='softmax', name='softmax')(time_rnn)
        loss = 'sparse_categorical_crossentropy'
        accuracy = ['sparse_categorical_accuracy']

    model = models.Model(inputs=[words_in], outputs=[output])
    return model, loss, accuracy


if __name__ == '__main__':
    args = get_arguments()
    ds_specs = loader.load_specs(args.ds)

    # Dataset specs
    MODE = ds_specs['mode']
    ALL_FILES = ds_specs['all_files']
    TRN_FILES = ds_specs['trn_files']
    DEV_FILES = ds_specs['dev_files']
    TST_FILES = ds_specs['tst_files']
    OTHER_CLASSES = ds_specs['other_classes']
    CLASS_MAP = ds_specs['class_map']
    STORE_ALL_VERSIONS = True

    # Some model hyperparameters
    EPOCHS = 30
    BDP_SEQ_SIZE = 15
    BDP_SEQ_OVERLAP = 0
    MAX_SENT_LEN = 50
    HIDDEN_SIZE = 30
    DROP_PROB = 0.5

    print('Embeddings: {}'.format(args.embeddings))
    print('Dataset mode: {}'.format(MODE))
    print('Other classes: "{}"'.format(', '.join(OTHER_CLASSES)))

    X, _ = loader.load_all_excel('../../data/' + ALL_FILES, merge=True, mode=MODE)
    X = pre.advanced_pre(X, None)

    embeddings = KeyedVectors.load(args.embeddings)
    embeddings.init_sims(replace=True)
    E, word2index, missing = loader.load_embeddings_matrix(embeddings, X, top_k=2000)
    vocab = set(word2index.keys())
    print(f'{missing} words have no embedding.')

    del X
    del missing

    pload = functools.partial(seq_utils.load_segments,
                              word2index=word2index,
                              sentence_len=MAX_SENT_LEN,
                              segment_size=BDP_SEQ_SIZE,
                              overlap=BDP_SEQ_OVERLAP,
                              vocab=vocab,
                              other_classes=OTHER_CLASSES,
                              class_map=CLASS_MAP,
                              mode=MODE)

    trn_x, trn_y, lencoder = pload(files=TRN_FILES)
    dev_x, dev_y, _        = pload(files=DEV_FILES, lencoder=lencoder)
    tst_x, tst_y, _        = pload(files=TST_FILES, lencoder=lencoder)
    
    print('Input features matrix shape: ', trn_x.shape)
    print('Amount of classes (+ PAD class): ', len(lencoder.classes_))

    all_acc = list()
    all_pre = list()
    all_rec = list()
    all_f1s = list()
    print('----')
    print(lencoder.classes_)
    for modelno in range(1, args.n_models + 1):
        print(f'--------------- << Training model [{modelno}/{args.n_models}] >> ---------------')
        print()

        model, loss, acc = build_model(
            sentence_len=MAX_SENT_LEN,
            segment_len=BDP_SEQ_SIZE,
            hidden_size=HIDDEN_SIZE,
            dropout_prob=DROP_PROB,
            n_classes=len(lencoder.classes_),
            crf_lambda=1e-3,
            use_crf=args.use_crf,
            use_context=args.use_context
        )

        model.summary()

        opt = optim.Adam()
        model.compile(opt, loss, acc)
        
        if STORE_ALL_VERSIONS:
            model_name = 'partial_{}'.format(datetime.now()) + '_epoch={epoch}.h5'
            model_folder = os.path.join(args.destination, model_name)
        else:
            model_name = 'full_model_{}.h5'.format(datetime.now())
            model_folder = os.path.join('../trained_models/', model_name)

        print('Storing partial models')
        print(f'model name: {model_name}')
        print(f'model location: {model_folder}')
        
        lr_scheduler = SGDRScheduler(min_lr=1e-4, max_lr=5e-3,
                                     steps_per_epoch=int(np.ceil(trn_x.shape[0] / 32)),
                                     mult_factor=2, cycle_length=2, lr_decay=0.95)
        training_history = model.fit(
            trn_x,
            trn_y,
            batch_size=32,
            epochs=EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[
                lr_scheduler,
                callbacks.ModelCheckpoint(filepath=model_folder, save_best_only=(not STORE_ALL_VERSIONS), verbose=1),
                callbacks.TensorBoard(log_dir='../../logs/{}'.format(model_name))
            ])

        # Evaluation part
        if STORE_ALL_VERSIONS:
            # If all epochs were stored, need to figure out which ones was the best
            min_loss = np.argmin(training_history.history['val_loss'])
            model_to_restore = model_folder.format(epoch=min_loss)
        else:
            model_to_restore = model_folder
        
        best_model = models.load_model(model_to_restore,
                                       custom_objects={'CRF':CRF,
                                                       'crf_loss': crf_loss,
                                                       'crf_viterbi_accuracy': crf_viterbi_accuracy,
                                                       'AttentionLayer': AttentionLayer}
                                      )

        # this accuracy already disregards 0-labels
        _, acc = best_model.evaluate(tst_x, tst_y)
        y_hat = best_model.predict(tst_x).argmax(-1)

        precision, recall, fscore = metrics.compute_fmetrics(tst_y.flatten(), y_hat.flatten(), lencoder)

        all_acc.append(acc)
        all_pre.append(precision)
        all_rec.append(recall)
        all_f1s.append(fscore)

        print(f'Model: {model_to_restore}')
        print('Test accuracy:  {:4.4f}'.format(acc))
        print('Test precision: {:4.4f}'.format(precision))
        print('Test recall:    {:4.4f}'.format(recall))
        print('Test F1:        {:4.4f}'.format(fscore))

        print('Current acc avg: {}'.format(np.average(all_acc)))
        print('Current f1s avg: {}'.format(np.average(all_f1s)))
        
        print('Ending epochs right before cycle restart: {}'.format(
            ', '.join(map(str, lr_scheduler.history['restart_epoch']))
        ))

    print('\n'*3)
    print('Overall results:')
    print('Acc =', end=' ')
    print(all_acc)
    print('Pre =', end=' ')
    print(all_pre)
    print('Rec =', end=' ')
    print(all_rec)
    print('F1  =', end=' ')
    print(all_f1s)
