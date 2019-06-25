from sklearn.metrics import precision_recall_fscore_support


def compute_fmetrics(y_true, y_hat, lencoder, average='macro'):
    """Computes MACRO precision, recall and f1 score ignoring the <PADL> label.
    :param y_true: The ground truth labels as a 1-d tensor
    :param y_hat: The predictions as a 1-d tensor
    :param instance of the label encoder
    :returns (precision, recall, f1) ignoring the <PADL> label
    """

    # all labels, but <PADL>, should be considered in the metrics
    pad_label_id = lencoder.classes_.tolist().index('<PADL>')
    allowed_labels = list(range(len(lencoder.classes_)))
    allowed_labels.remove(pad_label_id)

    # computing MACRO precision, recall, f1
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true,
                                                                   y_hat,
                                                                   average=average,
                                                                   labels=allowed_labels)
    return precision, recall, fscore
