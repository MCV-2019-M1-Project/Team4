import numpy as np

def maskEvaluation(annotation_mask, result_mask):
    true_positive = 0
    false_positive = 0
    false_negative = 0 
    true_negative = 0

    true_positive = np.sum(np.logical_and(annotation_mask == 255, result_mask == 255))     
    false_positive = np.sum(np.logical_and(result_mask == 255, annotation_mask != result_mask))
    false_negative = np.sum(np.logical_and(annotation_mask == 255, annotation_mask != result_mask))   
    
    precision = true_positive / (true_positive + false_positive)

    recall = true_positive / (true_positive + false_negative)

    F1_measure = 2 * ((precision * recall) / (precision + recall))
   
    return recall, precision, F1_measure
