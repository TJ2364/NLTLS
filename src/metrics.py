import numpy as np
from sklearn.metrics import roc_auc_score


def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def MRR(y_true, y_pred):
    index = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, index)
    score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(score) / np.sum(y_true)


def DCG(y_true, y_pred, n):
    index = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, index[:n])
    score = (2 ** y_true - 1) / np.log2(np.arange(len(y_true)) + 2)
    return np.sum(score)


def nDCG(y_true, y_pred, n):
    return DCG(y_true, y_pred, n) / DCG(y_true, y_true, n)
