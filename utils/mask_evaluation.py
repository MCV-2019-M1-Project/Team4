import numpy as np

def maskEvaluation(annotation_mask, result_mask):
    true_postive = 0
    false_positive = 0
    false_negative = 0 
    true_negative = 0

    if annotation_mask == result_mask == 1:
        true_postive =+ 1
    if (result_mask == 1 & annotation_mask != result_mask):
        false_positive =+ 1
    if (annotation_mask == 1 & annotation_mask != result_mask):
        false_negative =+ 1
    if (annotation_mask == result_mask == 0):
        true_negative =+ 1
    
    precision = true_positive / (true_postive + false_positive)

    recall = true_positive / (true_positive + false_negative)

    F1_measure = 2 * ((precision * recall) / (precision + recall))