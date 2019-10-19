"""
Usage:
  main.py <colorBase> <dimension> <query_set_path> <metric> <k> <background_removal> <text_removal> <text_removal_method>
Options:
"""

# VSCode imports
#from evaluation import *
#from mask import *

# PyCharm Imports
from week2.evaluation import *
from week2.mask import *

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
    text_method = int(sys.argv[8])

    # IMPORTANT PARAMETERS
    save_to_pickle = False
    save_to_pickle_text = False
    ground_truth_available = True
    ground_truth_text_available = True
    multiple_subimages = False
    level = 1

    # Get Ground Truth
    if ground_truth_available:
        print("Loading Ground Truth")
        GT = get_ground_truth(query_set_path + 'gt_corresps.pkl')
    
    if ground_truth_text_available:
        print("Loading Text Ground Truth")
        GT_text = get_ground_truth(query_set_path + 'text_boxes.pkl')

    # Get museum images filenames
    print("Getting Museum Images")
    museum_filenames = glob.glob(test_set_path + '*.jpg')
    museum_filenames.sort()

    # Get Museum Histograms
    print("Getting Museum Histograms")
    museum_histograms = {}
    idx = 0
    for museum_image in museum_filenames:
        print("Getting Histogram for Museum Image " + str(idx))
        museum_histograms[idx] = calculate_image_histogram(museum_image, None, color_base, dimension, level, None, None)
        idx += 1

    # Get query images filenames
    print("Getting Query Image")
    query_filenames = glob.glob(query_set_path + '*.jpg')
    query_filenames.sort()
    query_histograms = {}

    # Detect bounding boxes for text (result_text) and compute IoU parameter
    if text_removal == "True":
        print("Detecting text in the image")
        result_text = detect_bounding_boxes(query_set_path, text_method)

        # Check if the text results need to be saved in a pickle file
        if save_to_pickle_text:
            print("Saving Results to Pickle File")
            save_to_pickle_file(result_text, 'results/QST1/method2/text_boxes.pkl')

        # Evaluation of the text Removal
        IoU = evaluate_text(GT_text, result_text)
        print("Intersection over Union: ", str(IoU))

    # Get query images histograms
    print("Getting Query Histograms")
    idx = 0
    masks = {}
    for query_image in query_filenames:
        print("Getting Histogram for Query Image " + str(idx))
        if text_removal == "True" and not multiple_subimages:
            # Get histograms for each one of the images in the query filenames
            number_subimages = {}
            temp_img = cv2.imread(query_image)
            temp = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)
            temp[result_text[idx][0][0]:result_text[idx][0][2], result_text[idx][0][1]:result_text[idx][0][3]] = 255
            query_histograms[idx] = calculate_image_histogram(query_image, temp, color_base, dimension, level, None,
                                                              None)
        elif text_removal == "True" and multiple_subimages:
            print('Getting the QSD2_W2 background Masks')
            q_mask_filenames = glob.glob(query_set_path + '*.png')
            q_mask_filenames.sort()

            query_histograms = {}

            temp_img = cv2.imread(query_image)
            temp_img_sin_bck_text = cv2.imread(query_image.replace('.jpg', '_sin_bck_text.png'))
            # Get the text mask
            # Get the text mask
            temp = np.ones((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)
            for indx in range(0, len(result_text[idx])):
                temp[result_text[idx][indx][1]:result_text[idx][indx][3], result_text[idx][indx][0]:result_text[idx][indx][2]] = 0
            temp_text = np.concatenate((np.expand_dims(temp,axis=2), np.expand_dims(temp,axis=2), np.expand_dims(temp,axis=2)), axis=2)
            temp_sin_text = temp_img * temp_text

            # Get the background mask
            temp_bck_mask = cv2.imread(query_image.replace('.jpg', '.png'))
            temp_bck_mask = np.where(temp_bck_mask==255, 1, temp_bck_mask)

            output = paintings_detection(query_image.replace('.jpg', '_sin_text.png'), cv2.cvtColor(temp_bck_mask,
                                                                                                 cv2.COLOR_BGR2GRAY))
            if output > 0:
                number_subimages[idx] = 2
                query_histograms[idx] = calculate_image_histogram(query_image.replace('.jpg', '_1_sin_bck_text.png'),
                                                                  None, color_base, dimension, level, None, None)
                idx += 1
                query_histograms[idx] = calculate_image_histogram(query_image.replace('.jpg', '_2_sin_bck_text.png'),
                                                                  None, color_base, dimension, level,  None, None)
                idx += 1
            else:
                number_subimages[idx] = 1
                query_histograms[idx] = calculate_image_histogram(query_image.replace('.jpg', '_sin_bck_text.png'), None,
                                                                  color_base, dimension, level, None, None)
        elif background_removal == "True":
            masks[idx] = get_mask(query_image, masks_path, idx)
            # Detects if there is more than one painting (0 if there is only one painting)
            x_pixel_to_split = paintings_detection(query_image, masks[idx])
            if x_pixel_to_split == 0:  # Only one painting
                query_histograms[idx] = calculate_image_histogram(query_image, masks[idx], color_base, dimension, level,
                                                                  None, None)
        else:
            query_histograms[idx] = calculate_image_histogram(query_image, None, color_base, dimension, level, None,
                                                              None)
        idx += 1

    # Compute similarities to museum images for each image in the Query Set 1 and 2
    if multiple_subimages:
        print("Getting Similarities for Query Set2 and Museum")
        predictions = calculate_similarities(color_base, metric, dimension, query_histograms, museum_histograms)
        top_k = get_top_k(predictions, k, number_subimages)
    else:
        print("Getting Predictions")
        predictions = calculate_similarities(color_base, metric, dimension, query_histograms, museum_histograms)
        top_k = get_top_k(predictions, k, None)

    print("Ground Truth")
    print(GT)
    print("Top " + str(k))
    print(top_k)

    if save_to_pickle:
        print("Saving Results to Pickle File")
        save_to_pickle_file(top_k, 'results/QST1/method2/hypo_corresps.pkl')

    if ground_truth_available:
        map_k = get_mapk(GT, predictions, k)
        print('Map@K result: ' + str(map_k))
    
    if query_set_path == "qsd2_w1":   # Now crashes because of masks names, so I changed that
        print("Getting Precision, Recall and F1 score")
        GT_masks = glob.glob(query_set_path + '000*.png')  # Load masks from the ground truth
        GT_masks.sort()

        print(GT_masks)

        mean_precision = []
        mean_recall = []
        mean_f1score = []
        for idx, mask in masks.items():  # For each pair of masks, obtain the recall, precision and f1score metrics
            recall, precision, f1score = evaluate_mask(cv2.cvtColor(cv2.imread(GT_masks[idx]), cv2.COLOR_BGR2GRAY),
                                                       mask)
            mean_recall.append(recall)
            mean_precision.append(precision)
            mean_f1score.append(f1score)

        print('Recall: ' + str(np.array(mean_recall).mean()))
        print('Precision: ' + str(np.array(mean_precision).mean()))
        print('F1 score: ' + str(np.array(mean_f1score).mean()))
