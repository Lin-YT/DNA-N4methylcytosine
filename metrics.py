# standard imports
import numpy as np
import math

# third-party imports
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp+tn) / (tp+fp+fn+tn)
    sen = tp / (tp+fn)
    spe = tn / (tn+fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    return round(acc, 3), round(sen, 3), round(spe, 3), round(mcc, 3), tn, fp, fn, tp