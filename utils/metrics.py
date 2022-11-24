import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from ignite import metrics
from typing import Dict, List, Sequence

import warnings
warnings.filterwarnings("ignore")

"""
Adapted from @danikiyasseh
Source: https://github.com/danikiyasseh/CLOCS/blob/master/prepare_miscellaneous.py
"""

def evaluate_single(labels_list,outputs_list,classification="single"):
    if classification == "regression":
        return np.mean(np.abs(labels_list - outputs_list)), np.mean(np.square(labels_list - outputs_list))
    else:
        return calculate_auc(outputs_list,labels_list), calculate_acc(outputs_list,labels_list,classification=classification)


def calculate_auc(outputs_list,labels_list):
    """
    Multi-label binary classification or multi-class classification. 
    """
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    all_auc = []
    for i in range(labels_ohe.shape[1]):
        auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
        all_auc.append(auc)
    epoch_auroc = np.mean(all_auc)
    return epoch_auroc

def calculate_acc(outputs_list,labels_list,classification="single"):
    if classification == 'multilabel': 
        """ Convert Preds to Multi-Hot Vector """
        preds_list = np.where(outputs_list>0.5,1,0)
        """ Indices of Hot Vectors of Predictions """
        preds_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in preds_list]
        """ Indices of Hot Vectors of Ground Truth """
        labels_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in labels_list]
        """ What Proportion of Labels Did you Get Right """
        acc = np.array([np.isin(preds,labels).sum() for preds,labels in zip(preds_list,labels_list)]).sum()/(len(np.concatenate(preds_list)))        
    elif classification == "binary":
        preds_list = np.squeeze(np.where(outputs_list>0.5,1,0))
        ncorrect_preds = (preds_list == labels_list).sum().item()
        acc = ncorrect_preds/preds_list.shape[0]
    else: #normal single label setting 
        preds_list = torch.argmax(torch.tensor(outputs_list),1)
        ncorrect_preds = (preds_list == torch.tensor(labels_list)).sum().item()
        acc = ncorrect_preds/preds_list.shape[0]
    return acc

"""
Adapted from @YugenTen
Source: https://github.com/YugeTen/adios/blob/21c6f23f74656046f93105989874c7bac62cefa6/src/utils/metrics.py
"""

def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.
    """
    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value

"""
Adapted from roech_ecg
Source: https://github.com/DeepPSP/torch_ecg/blob/569e66c0182e9257217147cec9cf86e302813aff/benchmarks/train_crnn_cinc2021/scoring_metrics.py
"""

def compute_challenge_metric(
    weights: np.ndarray,
    labels: np.ndarray,
    outputs: np.ndarray,
    classes: List[str],
    sinus_rhythm: str,
) -> float:
    """ """
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError("The sinus rhythm class is not available.")

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(
            correct_score - inactive_score
        )
    else:
        normalized_score = 0.0

    return normalized_score

def compute_modified_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """
    Compute a binary multi-class, multi-label confusion matrix,
    where the rows are the labels and the columns are the outputs.
    """
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(
            max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1)
        )
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization

    return A