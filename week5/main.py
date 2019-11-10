"""
Usage:
  main.py <query_set_path> <background_removal> <text_removal> <text_removal_method> <k> 
Options:
"""

# VSCode imports
from evaluation import *
from mask import *
from texture_descriptors import *
from text_ocr import *
from compute_text_distances import *
from local_descriptors import *
from matching_distances import *
from painting_finder import *

# PyCharm Imports
# from week5.evaluation import *
# from week5.texture_descriptors import *
# from week5.text_ocr import *
# from week5.local_descriptors import extract_local_descriptors
# from week5.painting_finder import *

import sys
import glob
import numpy as np
import cv2
import os


if __name__ == '__main__':

    # read args
    print("Reading Arguments")
    query_set_path = 'images/' + sys.argv[1]
    test_set_path = 'images/bbdd/'
    masks_path = 'masks/'
    background_removal = sys.argv[2]
    text_removal = sys.argv[3]
    text_method = int(sys.argv[4])
    k = int(sys.argv[5])
    mask_gt_path = query_set_path
    mask_text_path = 'text/text_masks/'

    # GT and results parameters
    save_to_pickle = False
    save_to_pickle_text = False
    ground_truth_available = True
    ground_truth_text_available = True
    ground_truth_ocr_available = True

    # Denoise parameters
    execute_denoise_process = False
    use_denoised_images = True

    # Texture parameters
    texture_descriptors = True
    texture_descriptor_level = 3
    texture_method = "LBP"

    # Histogram parameters
    histogram_descriptors = True
    color_base = "LAB"
    dimension = '2D'
    metric = "bhattacharya_distance"
    level = 3

    # Text parameters
    text_descriptors = True

    # Local descriptors parameters
    local_descriptors = True
    local_method = "sift" # sift, surf, root_sift, orb, fast-daisy, brisk
    matching_method = "flann" # brute_force, flann, nmslib
    local_metric = "l2" # l1, l2, hamming, hamming2
    matches_threshold = 5

    multiple_subimages = True

    # Get Ground Truth
    if ground_truth_available:
        print("Loading Ground Truth")
        GT = get_ground_truth(query_set_path + '/' + 'gt_corresps.pkl')

    if ground_truth_text_available:
        print("Loading Text Ground Truth")
        GT_text = get_ground_truth(query_set_path + '/' + 'text_boxes.pkl')

    # Get museum images filenames
    print("Getting Museum Images")
    museum_filenames = glob.glob(test_set_path + '*.jpg')
    museum_filenames.sort()
    number_museum_elements = len(museum_filenames)

    # Get Museum Histograms
    print("Getting Museum Histograms")

    # Check the data structures needed to store the features
    if histogram_descriptors:
        museum_histograms = {}
    else:
        museum_histograms = None
    if texture_descriptors:
        museum_textures = {}
    else:
        museum_textures = None
    if text_descriptors:
        museum_text_gt = {}
        museum_text_gt_filenames = glob.glob('text/bbdd_text/*.txt')
        museum_text_gt_filenames.sort()
        mask_text_filenames = glob.glob('text/' + '*.txt')
        for txt_file in mask_text_filenames:
            os.remove(txt_file)
    else:
        museum_text_gt = None
    if local_descriptors:
        museum_local_descriptors = {}
    else:
        museum_local_descriptors = None

    idx = 0
    for museum_image in museum_filenames:
        print("Getting Features for Museum Image " + str(idx))

        # Get histogram for museum image
        if histogram_descriptors:
            museum_histograms[idx] = calculate_image_histogram(museum_image, None, color_base, dimension, level, None,
                                                               None)

        # Get texture descriptor for museum image
        if texture_descriptors:
            museum_textures[idx] = get_image_texture_descriptor(museum_image, texture_method, texture_descriptor_level,
                                                                None, None, None)

        # Get text descriptor for museum image.
        if text_descriptors:
            # Read GT for BBDD text_files
            if ground_truth_ocr_available:
                with open(museum_text_gt_filenames[idx], 'r') as file:
                    line = file.readline()
                    if not line:
                        museum_text_gt[idx] = line
                    else:
                        line = line.split(',')
                        museum_text_gt[idx] = line[0][2:-1]

        # Get local descriptor for museum image
        if local_descriptors:
            museum_local_descriptors[idx] = extract_local_descriptors(museum_image, None, local_method, None, None)

        idx += 1

    # Remove noise from query set images and save the denoised images in a new folder used for the pipeline
    if execute_denoise_process:
        query_noise_filenames = glob.glob(query_set_path + '/' + '*.jpg')
        query_noise_filenames.sort()
        idx = 23
        PSNR = []  # Peak signal to Noise Ratio
        for query_noise_image in query_noise_filenames:
            if ground_truth_available:
                PSNR = remove_noise(test_set_path, query_set_path, query_noise_image, GT, idx, PSNR)
            else:
                PSNR = remove_noise(test_set_path, query_set_path, query_noise_image, 0, idx, PSNR)
            idx += 1

        print("Minimum Peak Signal to Noise Ratio: " + str(np.min(PSNR)))
    
    # Use denoised images or not
    if use_denoised_images:
        # Get query images filenames
        query_set_denoised_path = query_set_path + '_denoised/'
        query_filenames = glob.glob(query_set_denoised_path + '*.jpg')
        query_filenames.sort()
    else:
        # Get query images filenames
        print("Getting Query Image")
        query_filenames = glob.glob(query_set_path + '/' + '*.jpg')
        query_filenames.sort()


    # Get query images histograms
    print("Getting Query Features")
    idx = 0
    masks = {}
    cropped_images = {}
    paintings_data = []
    number_query_elements = len(query_filenames)
    result_text = []


    # Check the data structures needed to store the features
    if histogram_descriptors:
        query_histograms = {}
    else:
        query_histograms = None
    if texture_descriptors:
        query_textures = {}
    else:
        query_textures = None
    if text_descriptors:
        query_ocrs = {}
        query_text_gt = {}
    else:
        query_ocrs = None
    if local_descriptors:
        query_local_descriptors = {}
    else:
        query_local_descriptors = None

    if text_removal and multiple_subimages:
        number_subimages = {}
        query_features_idx = 0

    # Iterate over the query images to extract the features
    for query_image in query_filenames:
        print("Finding subpaintings in Query Image " + str(idx))

        if text_removal and multiple_subimages:

            # Get background masks
            masks[idx], cropped_paintings, query_painting_data, number_subpaintings = find_paintings(query_image, masks_path, idx, query_set_path)
            paintings_data.append(query_painting_data)
            number_subimages[idx] = number_subpaintings

            idx += 1
    
    if text_removal and multiple_subimages:
        query_cropped_filenames = glob.glob(query_set_path + '_cropped_images/' + '*.jpg')
        query_cropped_filenames.sort()   
        query_features_idx = 0

    for query_cropped_image in query_cropped_filenames:  
        print("Getting features for Query Image " + str(query_features_idx)) 

        result_text.extend(detect_bounding_boxes(query_cropped_image, mask_text_path, text_method, True, False, query_features_idx))
        text_mask = cv2.cvtColor(cv2.imread(query_cropped_image),cv2.COLOR_BGR2GRAY)
        text_mask[:,:] = 255
        text_mask[result_text[query_features_idx][0][1]:result_text[query_features_idx][0][3], result_text[query_features_idx][0][0]:result_text[query_features_idx][0][2]] = 0

        if histogram_descriptors:
            query_histograms[query_features_idx] = calculate_image_histogram(query_cropped_image, text_mask,
                                                                                color_base, dimension, level,
                                                                                None, None)

        if texture_descriptors:
            query_textures[query_features_idx] = get_image_texture_descriptor(query_cropped_image, texture_method,
                                                                                texture_descriptor_level,
                                                                                text_mask, None, None)

        if text_descriptors:
            query_ocrs[query_features_idx] = get_text(query_cropped_image, mask_text_path, text_method, query_features_idx, None,
                                                        None, True)

        if local_descriptors:
            query_local_descriptors[query_features_idx] = extract_local_descriptors(query_cropped_image, text_mask,
                                                                                    local_method, None, None)
        
        query_features_idx += 1


    # Check if the text results need to be saved in a pickle file
    if save_to_pickle_text:
        print("Saving Results to Pickle File")
        save_to_pickle_file(result_text, 'results/QST1/method2/text_boxes.pkl')

    # Evaluation of the text Removal
    if ground_truth_text_available:
        IoU = evaluate_text(GT_text, result_text)
        print("Intersection over Union: ", str(IoU))

    # Compute similarities to museum images for each image
    if multiple_subimages:
        print("Getting Similarities for Query Set and Museum")
        print("Number of paintings in query set when there are subpaintings: ", query_features_idx)
        predictions = calculate_similarities(color_base, metric, dimension, query_histograms, query_textures,
                                             query_ocrs, query_local_descriptors, museum_histograms, museum_textures, museum_text_gt,
                                             museum_local_descriptors, query_features_idx, number_museum_elements,
                                             matching_method, local_metric, matches_threshold)
        top_k = get_top_k(predictions, k, number_subimages)
    else:
        print("Getting Predictions")
        predictions = calculate_similarities(color_base, metric, dimension, query_histograms, query_textures,
                                             query_ocrs, query_local_descriptors, museum_histograms, museum_textures, museum_text_gt,
                                             museum_local_descriptors, number_query_elements, number_museum_elements,
                                             matching_method, local_metric, matches_threshold)
        top_k = get_top_k(predictions, k, None)

    if ground_truth_available:
        print("Ground Truth")
        print(GT)
        print("Top " + str(k))
        print(top_k)

    if save_to_pickle:
        print("Saving Results to Pickle File")
        save_to_pickle_file(top_k, 'results/QST1/method2/hypo_corresps.pkl')

    if ground_truth_available:
        map_k = get_mapk(GT, top_k, k)
        print('Map@K result: ' + str(map_k))

    if background_removal == "True":
        print("Getting Precision, Recall and F1 score")
        GT_masks = glob.glob(mask_gt_path + '/000*.png')  # Load masks from the ground truth
        GT_masks.sort()

        mean_precision = []
        mean_recall = []
        mean_f1score = []
        for idx, mask in masks.items():  # For each pair of masks, obtain the recall, precision and f1score metrics
            recall, precision, f1score = evaluate_mask(cv2.cvtColor(cv2.imread(GT_masks[idx]), cv2.COLOR_BGR2GRAY), mask, idx)
            mean_recall.append(recall)
            mean_precision.append(precision)
            mean_f1score.append(f1score)

        print('Precision: ' + str(np.array(mean_precision).mean()))
        print('Recall: ' + str(np.array(mean_recall).mean()))
        print('F1 score: ' + str(np.array(mean_f1score).mean()))
