"""
Usage:
  main.py <colorBase> <querySetPath> <metric> <k> <backgroundRemoval>
Options:
"""

from week1 import evaluation, mask_evaluation
import sys
import glob
import numpy as np
import cv2


if __name__ == '__main__':

    # read args
    colorBase = sys.argv[1]
    querySetPath = sys.argv[2]
    metric = sys.argv[3]
    k = sys.argv[4]
    backgroundRemoval = sys.argv[5]

    mask_apply = False
    save_to_pickle = False

    # Remove background for each image in the Query Set 2
    if backgroundRemoval == "True":
        mask_apply = evaluation.get_mask('images/' + querySetPath + '/')
      
    # Select images depending on if they have background remove or not
    if mask_apply:
        query_set_path = querySetPath + '/' + '*_image_with_mask.png' # Images without background
    else:
        query_set_path = querySetPath + '/' + '*.jpg' # Images with background

    GT = evaluation.get_ground_truth('images/' + querySetPath + '/gt_corresps.pkl')
    DB_Histograms = evaluation.calculate_image_histograms('images/bbdd/*.jpg', colorBase)
    QS_Histograms = evaluation.calculate_image_histograms('images/' + query_set_path, colorBase)

    # Compute similarities to museum images for each image in the Query Set 1 and 2
    predictions = evaluation.calculate_similarities(colorBase, metric, QS_Histograms, DB_Histograms)
    top_k = evaluation.get_top_k(predictions, int(k))

    if save_to_pickle:
        evaluation.save_to_pickle_file(top_k, '../results/QST1/method2/hypo_corresps.pkl')
    map_k = evaluation.get_mapk(GT, predictions, int(k))

    print('Map@K result: ' + str(map_k))
    
    if mask_apply:

        GT_mask = glob.glob('../images/'+querySetPath+'/000*.png')  # Load maks from the ground truth
        GT_mask.sort()
        P_mask = glob.glob('../images/'+querySetPath+'/a*_mask.png')  # Load masks obtained before
        P_mask.sort()
        
        mean_precision = []
        mean_recall = []
        mean_f1score = []
        
        for i in range(0, len(GT_mask)): # For each pair of masks, obtain the recall, precision and f1score metrics
            recall, precision, f1score = mask_evaluation.mask_evaluation(cv2.cvtColor(cv2.imread(GT_mask[i]),
                                                                                      cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.imread(P_mask[i]), cv2.COLOR_BGR2GRAY))
            mean_recall.append(recall)
            mean_precision.append(precision)
            mean_f1score.append(f1score)

        print('Recall: ' + str(np.array(mean_recall).mean()))
        print('Precision: ' + str(np.array(mean_precision).mean()))
        print('F1 score: ' + str(np.array(mean_f1score).mean()))
