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

    GT = evaluation.get_ground_truth('../images/' + querySetPath + '/gt_corresps.pkl')
    DB_Histograms = evaluation.calculate_image_histograms('../images/bbdd/', colorBase)
    QS_Histograms = evaluation.calculate_image_histograms('../images/' + querySetPath + '/', colorBase)

    # Compute similarities to museum images for each image in the Query Set 1
    predictions = evaluation.calculate_similarities(colorBase, metric, QS_Histograms, DB_Histograms)
    top_k = evaluation.get_top_K(predictions, int(k))
    map_k = evaluation.get_mapk(GT, predictions, int(k))

    print('Map@K result: ' + str(map_k))
