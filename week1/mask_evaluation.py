import numpy as np


def mask_evaluation(annotation_mask, result_mask):
    """
    This function calculates the Precision, Recall and F1 score by comparing the ground truth mask
    with the mask obtained with our algorithm.

    :param annotation_mask: ground truth maks
    :param result_mask: obtained masks
    :return: precision, recall and F1 score
    """

    true_positive = np.sum(np.logical_and(annotation_mask == 255, result_mask == 255))     
    false_positive = np.sum(np.logical_and(result_mask == 255, annotation_mask != result_mask))
    false_negative = np.sum(np.logical_and(annotation_mask == 255, annotation_mask != result_mask))   
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_measure = 2 * ((precision * recall) / (precision + recall))
   
    return recall, precision, f1_measure
