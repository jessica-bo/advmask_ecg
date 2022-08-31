import pickle
import os
import torch.nn as nn
import torch
import math
import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import wandb
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple, Union


def simclr_loss_fn(latent_embeddings, pids=None, positive_pairing="SimCLR", temperature=0.1):
    """ Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        latent_embeddings (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN)
        pids (list): patient ids of instances in batch
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    """
    # if positive_pairing in ['CMSC','CMLC','CMSMLC']:
    #     pids = np.array(pids,dtype=np.object)   
    #     pid1,pid2 = np.meshgrid(pids,pids)
    #     pid_matrix = pid1 + '-' + pid2
    #     pids_of_interest = np.unique(pids + '-' + pids) #unique combinations of pids of interest i.e. matching
    #     bool_matrix_of_interest = np.zeros((len(pids),len(pids)))
    #     for pid in pids_of_interest:
    #         bool_matrix_of_interest += pid_matrix == pid
        # rows1,cols1 = np.where(np.triu(bool_matrix_of_interest,1))
        # rows2,cols2 = np.where(np.tril(bool_matrix_of_interest,-1))

    loss = 0
    view1_array = latent_embeddings[0] #(BxH)
    view2_array = latent_embeddings[1] #(BxH)
    norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
    norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
    sim_matrix = torch.mm(view1_array,view2_array.transpose(0,1))
    norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)
    argument = sim_matrix/(norm_matrix*temperature)
    sim_matrix_exp = torch.exp(argument)
    
    if positive_pairing in ['SimCLR', 'CMSC']:
        self_sim_matrix1 = torch.mm(view1_array,view1_array.transpose(0,1))
        self_norm_matrix1 = torch.mm(norm1_vector.transpose(0,1),norm1_vector)
        argument = self_sim_matrix1/(self_norm_matrix1*temperature)
        self_sim_matrix_exp1 = torch.exp(argument)
        self_sim_matrix_off_diagonals1 = torch.triu(self_sim_matrix_exp1,1) + torch.tril(self_sim_matrix_exp1,-1)
        
        self_sim_matrix2 = torch.mm(view2_array,view2_array.transpose(0,1))
        self_norm_matrix2 = torch.mm(norm2_vector.transpose(0,1),norm2_vector)
        argument = self_sim_matrix2/(self_norm_matrix2*temperature)
        self_sim_matrix_exp2 = torch.exp(argument)
        self_sim_matrix_off_diagonals2 = torch.triu(self_sim_matrix_exp2,1) + torch.tril(self_sim_matrix_exp2,-1)

        denominator_loss1 = torch.sum(sim_matrix_exp,1) + torch.sum(self_sim_matrix_off_diagonals1,1)
        denominator_loss2 = torch.sum(sim_matrix_exp,0) + torch.sum(self_sim_matrix_off_diagonals2,0)
        
        diagonals = torch.diag(sim_matrix_exp)
        loss_term1 = -torch.mean(torch.log(diagonals/denominator_loss1))
        loss_term2 = -torch.mean(torch.log(diagonals/denominator_loss2))
        loss += loss_term1 + loss_term2
        loss_terms = 2

    # elif positive_pairing in ['CMSC','CMLC','CMSMLC']: #ours #CMSMLC = positive examples are same instance and same patient
    #     # triu_elements = sim_matrix_exp[rows1,cols1]
    #     # tril_elements = sim_matrix_exp[rows2,cols2]
    #     diag_elements = torch.diag(sim_matrix_exp)
        
    #     triu_sum = torch.sum(sim_matrix_exp,1)
    #     tril_sum = torch.sum(sim_matrix_exp,0)
        
    #     loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
    #     loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
        
    #     # loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
    #     # loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))
        
    #     loss = loss_diag1 + loss_diag2
    #     loss_terms = 2

    #     # if len(rows1) > 0:
    #     #     loss += loss_triu #technically need to add 1 more term for symmetry
    #     #     loss_terms += 1
        
    #     # if len(rows2) > 0:
    #     #     loss += loss_tril #technically need to add 1 more term for symmetry
    #     #     loss_terms += 1
        
    loss = loss/(loss_terms)
    return loss

def obtain_neg_contrastive_loss(latent_embeddings, pids, trial, mask, ratio):
    contrastive_loss = -simclr_loss_fn(latent_embeddings, pids, trial)
    weight = 1
    sparsity_loss = 0

    for sample in mask:
        sparsity = 1/torch.sin(ratio * math.pi * torch.sum(sample) / torch.numel(sample))
        sparsity_loss = sparsity_loss + sparsity
    
    # sparsity_loss = 1/torch.sin(ratio * math.pi * torch.sum(mask) / torch.numel(mask))
    loss = contrastive_loss + weight * sparsity_loss
    return loss


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