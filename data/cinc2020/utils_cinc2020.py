from typing import Union, Optional, List, Sequence, Dict, Tuple
from numbers import Real

import numpy as np
from functools import reduce

from torch_ecg.databases.aux_data.cinc2020_aux_data import load_weights

def ensure_siglen(
    values: Sequence[Real],
    siglen: int,
    fmt: str = "lead_first",
    tolerance: Optional[float] = None,
) -> np.ndarray:
    """finished, checked,
    ensure the (ECG) signal to be of length `siglen`,
    strategy:
        If `values` has length greater than `siglen`,
        the central `siglen` samples will be adopted;
        otherwise, zero padding will be added to both sides.
        If `tolerance` is given,
        then if the length of `values` is longer than `siglen` by more than `tolerance`,
        the `values` will be sliced to have multiple of `siglen` samples.
    Parameters
    ----------
    values: sequence,
        values of the `n_leads`-lead (ECG) signal
    siglen: int,
        length of the signal supposed to have
    fmt: str, default "lead_first", case insensitive,
        format of the input and output values, can be one of
        "lead_first" (alias "channel_first"), "lead_last" (alias "channel_last")
    Returns
    -------
    out_values: ndarray,
        ECG signal in the format of `fmt` and of fixed length `siglen`,
        of ndim=3 if `tolerence` is given, otherwise ndim=2
    """
    if fmt.lower() in ["channel_last", "lead_last"]:
        _values = np.array(values).T
    else:
        _values = np.array(values).copy()
    original_siglen = _values.shape[1]
    n_leads = _values.shape[0]

    if tolerance is None or original_siglen <= siglen * (1 + tolerance):
        if original_siglen >= siglen:
            start = (original_siglen - siglen) // 2
            end = start + siglen
            out_values = _values[..., start:end]
        else:
            pad_len = siglen - original_siglen
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            out_values = np.concatenate(
                [
                    np.zeros((n_leads, pad_left)),
                    _values,
                    np.zeros((n_leads, pad_right)),
                ],
                axis=1,
            )

        if fmt.lower() in ["channel_last", "lead_last"]:
            out_values = out_values.T
        if tolerance is not None:
            out_values = out_values[np.newaxis, ...]

        return out_values

    forward_len = int(round(siglen * tolerance))
    out_values = np.array(
        [
            _values[..., idx * forward_len : idx * forward_len + siglen]
            for idx in range((original_siglen - siglen) // forward_len + 1)
        ]
    )
    if fmt.lower() in ["channel_last", "lead_last"]:
        out_values = np.moveaxis(out_values, 1, -1)
    return out_values



def list_sum(lst: Sequence[list]) -> list:
    """finished, checked,
    Parameters
    ----------
    lst: sequence of list,
        the sequence of lists to obtain the summation
    Returns
    -------
    l_sum: list,
        sum of `lst`,
        i.e. if lst = [list1, list2, ...], then l_sum = list1 + list2 + ...
    """
    l_sum = reduce(lambda a, b: a + b, lst, [])
    return l_sum

def remove_spikes_naive(sig: np.ndarray) -> np.ndarray:
    """finished, checked,
    remove `spikes` from `sig` using a naive method proposed in entry 0416 of CPSC2019
    `spikes` here refers to abrupt large bumps with (abs) value larger than 20 mV,
    or nan values (read by `wfdb`),
    do NOT confuse with `spikes` in paced rhythm
    Parameters
    ----------
    sig: ndarray,
        single-lead ECG signal with potential spikes
    Returns
    -------
    filtered_sig: ndarray,
        ECG signal with `spikes` removed
    """
    b = list(
        filter(
            lambda k: k > 0,
            np.argwhere(np.logical_or(np.abs(sig) > 20, np.isnan(sig))).squeeze(-1),
        )
    )
    filtered_sig = sig.copy()
    for k in b:
        filtered_sig[k] = filtered_sig[k - 1]
    return filtered_sig

class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__ methods.
    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """ """
        return []


def evaluate_12ECG_score(
    truth: Sequence, scalar_pred: Sequence
) -> Tuple[float]:
    """
    Parameters
    ----------
    classes: list of str,
        list of all the classes, in the format of abbrevations
    truth: sequence,
        ground truth array, of shape (n_records, n_classes), with values 0 or 1
    binary_pred: sequence,
        binary predictions, of shape (n_records, n_classes), with values 0 or 1
    scalar_pred: sequence,
        probability predictions, of shape (n_records, n_classes), with values within [0,1]
    Returns
    -------
    auroc: float,
    auprc: float,
    accuracy: float,
    f_measure: float,
    f_beta_measure: float,
    g_beta_measure: float,
    challenge_metric: float,
    """
    # # normal_class = '426783006'
    # normal_class = "NSR"
    # # equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    # classes = ['AF', 'AFL', 'IAVB', 'IRBBB', 'LAnFB', 'LBBB', 'LPR', 'LQT', 'NSIVCB', 'NSR', 'PAC', 'PVC', 'QAb', 'RBBB', 'SA', 'SB', 'STach', 'TAb', 'TInv']
    # weights = load_weights(classes=classes)

    _truth = np.array(truth)
    _binary_pred = (np.array(scalar_pred) > 0.5).astype(float)
    _scalar_pred = np.array(scalar_pred)

    auroc, auprc = compute_auc(_truth, _scalar_pred)
    accuracy = compute_accuracy(_truth, _binary_pred)
    # f_measure = compute_f_measure(_truth, _binary_pred)

    # Return the results.
    return (
        auroc,
        # auprc,
        accuracy,
        # f_measure,
    )

# Compute recording-wise accuracy.
def compute_accuracy(labels: np.ndarray, outputs: np.ndarray) -> float:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute macro F-measure.
def compute_f_measure(labels: np.ndarray, outputs: np.ndarray) -> float:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure

# Compute confusion matrices.
def compute_confusion_matrices(
    labels: np.ndarray, outputs: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """checked,"""
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1.0 / normalization
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")

    return A

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, float]:
    """checked,"""
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return macro_auroc, macro_auprc