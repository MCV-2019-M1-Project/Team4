import numpy as np

def maskEvaluation(annotation_mask, result_mask):
    true_positive = 0
    false_positive = 0
    false_negative = 0 
    true_negative = 0

    if annotation_mask.all() == result_mask.all() == 1:
        true_positive =+ 1
    if (result_mask.all() == 1 & annotation_mask.all() != result_mask.all()):
        false_positive =+ 1
    if (annotation_mask.all() == 1 & annotation_mask.all() != result_mask.all()):
        false_negative =+ 1
    if (annotation_mask.all() == result_mask.all() == 0):
        true_negative =+ 1
    
    precision = true_positive / (true_positive + false_positive)

    recall = true_positive / (true_positive + false_negative)

    F1_measure = 2 * ((precision * recall) / (precision + recall))
   
    return recall, precision, F1_measure
