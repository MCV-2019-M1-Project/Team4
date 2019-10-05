"""
Usage:
  main.py <colorBase> <querySetPath> <metric> <k> <backgroundRemoval>
Options:
"""
from utils import evaluation
import sys

if __name__ == '__main__':

    # read args
    colorBase = sys.argv[1]
    querySetPath = sys.argv[2]
    metric = sys.argv[3]
    k = sys.argv[4]
    backgroundRemoval = sys.argv[5]

    mask_apply = False

    # Remove background for each image in the Query Set 2
    if backgroundRemoval == "True":
        mask_apply = evaluation.get_mask('../images/' + querySetPath + '/')
      
    # Select images depending on if they have background remove or not
    if mask_apply:
        query_set_path = querySetPath + '/' + '*_image_with_mask.png' # Images without background
    else:
        query_set_path = querySetPath + '/' + '*.jpg' # Images with background

    GT = evaluation.get_ground_truth('../images/' + querySetPath + '/gt_corresps.pkl')
    DB_Histograms = evaluation.calculate_image_histograms('../images/bbdd/*.jpg', colorBase)
    QS_Histograms = evaluation.calculate_image_histograms('../images/' + query_set_path, colorBase)      
    
    # Compute similarities to museum images for each image in the Query Set 1 and 2
    predictions = evaluation.calculate_similarities(colorBase, metric, QS_Histograms, DB_Histograms)
    top_k = evaluation.get_top_k(predictions, int(k))
    map_k = evaluation.get_mapk(GT, predictions, int(k))

    print('Map@K result: ' + str(map_k))
