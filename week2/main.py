"""
Usage:
  main.py <colorBase> <dimension> <query_set_path> <metric> <k> <background_removal> <text_removal>
Options:
"""

from week2 import evaluation 
import sys
import glob
import numpy as np
import cv2

if __name__ == '__main__':

    # read args
    print("Reading Arguments")
    color_base = sys.argv[1]
    dimension = sys.argv[2]
    query_set_path = 'images/' + sys.argv[3] + '/'
    test_set_path = 'images/bbdd/'
    masks_path = 'masks/'
    metric = sys.argv[4]
    k = int(sys.argv[5])
    background_removal = sys.argv[6]
    text_removal = sys.argv[7]

    save_to_pickle = False
    save_to_pickle_text = False
    ground_truth_available = True
    ground_truth_text_available = True
    level = 2

    # Get Ground Truth
    if ground_truth_available:
        print("Loading Ground Truth")
        GT = evaluation.get_ground_truth(query_set_path + 'gt_corresps.pkl')
    
    if ground_truth_text_available:
        print("Loading Text Ground Truth")
        GT_text = evaluation.get_ground_truth(query_set_path + 'text_boxes.pkl')

    # Get museum images filenames
    print("Getting Museum Images")
    museum_filenames = glob.glob(test_set_path + '*.jpg')
    museum_filenames.sort()

     # Get query images filenames
    print("Getting Query Image")
    query_filenames = glob.glob(query_set_path + '*.jpg')
    query_filenames.sort()

    # Detect bounding boxes for text (result_text) and compute IoU parameter
    if text_removal:
        print("Detecting text in the image")
        result_text = evaluation.detect_bounding_boxes(query_set_path)
        IoU = evaluation.evaluate_text(GT_text, result_text)
        print("Intersection over Union: ", str(IoU))

    if save_to_pickle_text:
        print("Saving Results to Pickle File")
        evaluation.save_to_pickle_file(result_text, 'results/QST1/method2/text_boxes.pkl')

    # Get Museum Histograms
    print("Getting Museum Histograms")
    museum_histograms = {}
    idx = 0
    for museum_image in museum_filenames:
        print("Getting Histogram for Museum Image " + str(idx))
        museum_histograms[idx] = evaluation.calculate_image_histogram(museum_image, None,
                                                                      color_base, dimension, level)
        idx += 1

    # Get query images histograms
    print("Getting Query Histograms")
    idx = 0
    query_histograms = {}
    for query_image in query_filenames:
        masks = {}
        print("Getting Histogram for Query Image " + str(idx))
        if background_removal == "True":
            masks[idx] = evaluation.get_mask(query_image, masks_path, idx)
            query_histograms[idx] = evaluation.calculate_image_histogram(query_image,
                                                                         masks[idx],
                                                                         color_base, dimension, level)
        else:
            query_histograms[idx] = evaluation.calculate_image_histogram(query_image, None,
                                                                         color_base, dimension, level)
        idx += 1

    # Compute similarities to museum images for each image in the Query Set 1 and 2
    print("Getting Predictions")
    predictions = evaluation.calculate_similarities(color_base, metric, dimension, query_histograms, museum_histograms)
    top_k = evaluation.get_top_k(predictions, k)
    print("Ground Truth")
    print(GT)
    print("Top " + str(k))
    print(top_k)

    if save_to_pickle:
        print("Saving Results to Pickle File")
        evaluation.save_to_pickle_file(top_k, 'results/QST1/method2/hypo_corresps.pkl')

    if ground_truth_available:
        map_k = evaluation.get_mapk(GT, predictions, k)
        print('Map@K result: ' + str(map_k))
    
    if background_removal == "True":
        print("Getting Precision, Recall and F1 score")
        GT_masks = glob.glob(query_set_path + '000*.png')  # Load masks from the ground truth
        GT_masks.sort()

        mean_precision = []
        mean_recall = []
        mean_f1score = []

        for idx, mask in masks.items():  # For each pair of masks, obtain the recall, precision and f1score metrics
            recall, precision, f1score = evaluation.evaluate_mask(cv2.cvtColor(cv2.imread(GT_masks[idx]),
                                                                                      cv2.COLOR_BGR2GRAY),
                                                                         mask)
            mean_recall.append(recall)
            mean_precision.append(precision)
            mean_f1score.append(f1score)

        print('Recall: ' + str(np.array(mean_recall).mean()))
        print('Precision: ' + str(np.array(mean_precision).mean()))
        print('F1 score: ' + str(np.array(mean_f1score).mean()))
