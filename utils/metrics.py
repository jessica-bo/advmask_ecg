import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Sequence

import warnings
warnings.filterwarnings("ignore")

def evaluate_single(labels_list,outputs_list,classification="normal"):
    return calculate_auc(outputs_list,labels_list),calculate_acc(outputs_list,labels_list,classification=classification)

def calculate_auc(outputs_list,labels_list):
    """
    Multi-label binary classification or multi-class classification. 
    """
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    all_auc = []
    for i in range(labels_ohe.shape[1]):
        auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
        # print("auc {} of {}".format(auc, i))
        all_auc.append(auc)
    epoch_auroc = np.mean(all_auc)
    return epoch_auroc

def calculate_acc(outputs_list,labels_list,classification="normal"):
    if classification == 'multilabel': 
        """ Convert Preds to Multi-Hot Vector """
        preds_list = np.where(outputs_list>0.5,1,0)
        """ Indices of Hot Vectors of Predictions """
        preds_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in preds_list]
        """ Indices of Hot Vectors of Ground Truth """
        labels_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in labels_list]
        """ What Proportion of Labels Did you Get Right """
        acc = np.array([np.isin(preds,labels).sum() for preds,labels in zip(preds_list,labels_list)]).sum()/(len(np.concatenate(preds_list)))        
    else: #normal single label setting 
        preds_list = torch.argmax(torch.tensor(outputs_list),1)
        ncorrect_preds = (preds_list == torch.tensor(labels_list)).sum().item()
        acc = ncorrect_preds/preds_list.shape[0]
    return acc

def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.
    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.weighted_mean
        batch_size_key (str): key of batch size values.
    Returns:
        float: weighted mean of the values of a key
    """
    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value


class Entropy(nn.Module):
    """
    Computes the entropy.
    """
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return b.sum()