import matplotlib.pyplot as plt
import numpy as np

PREDICTIONS_FOLDER = '../data/predictions/'
DATA_FOLDER = '../data/'


def plot_graphs(history, metric):
    """
    Plots a metric (like loss or accuracy) on the training set against the validation set

    @param history: an object containing the metric values
    @param metric: the selected metric
    @return: None
    """
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


def get_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
