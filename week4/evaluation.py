import pickle

from distances_metrics import *
from histogram import *
from mask import *
from text import *
from compute_text_distances import *
from matching_distances import *
import glob
import ml_metrics as metrics
import numpy as np
import cv2


def get_ground_truth(path):
    """
    Returns the ground truth stored in a pickle file as a list of lists

    :param path: path where the ground truth is located
    :return: ground truth as a list of lists of stringss
    """

    pkl_file = open(path, 'rb')
    return pickle.load(pkl_file, fix_imports=True, encoding='ASCII', errors='strict')


def save_to_pickle_file(data_to_save, path):
    """
    This function saves given data into a pickle file
    :param data_to_save: object that wants to be saved
    :param path: path where the object will be saved
    :return: void
    """

    print(data_to_save)

    with open(path, 'wb') as handle:
        pickle.dump(data_to_save, handle)

    check = get_ground_truth(path)

    if check == data_to_save:
        print('Pickle File saved correctly')


def calculate_hist_distance(color_base, metric, dimension, hist_a, hist_b):
    """
    This function calculates the distance between 2 histograms in the chosen metric.

    :param color_base: string indicating the color base in which the histogram has been calculated
    :param dimension:
    :param metric: string indicating which metric to use to calculate the distance
    :param hist_a: array or matrix containing histogram 1
    :param hist_b: array or matrix containing histogram 2
    :return: float indicating the distance between the two histograms
    """

    distance = 0
    if color_base != "Grayscale" and dimension == 1:
        bins = 256
        aux_a = np.reshape(hist_a, (3, bins))
        aux_b = np.reshape(hist_b, (3, bins))
        for i in range(3):
            distance += compute_distance(aux_a[i], aux_b[i], metric)
    else:
        distance = compute_distance(hist_a, hist_b, metric)

    return distance


def calculate_text_distance(str_1, str_2, method):
    """

    :param str_1:
    :param str_2:
    :param method:
    :return:
    """

    if method == 'levenshtein':
        return levenshtein_distance(str_1, str_2)
    elif method == 'hamming':
        return hamming_distance(str_1, str_2)
    elif method == 'jaro_winkler':
        return jaro_winkler_distance(str_1, str_2)
    else:
        raise Exception("Wrong distance method")


def calculate_similarities(color_base, metric, dimension, query_hists, query_textures, query_ocrs, query_local_descriptors, 
                           museum_hists, museum_textures, museum_ocrs, museum_local_descriptors, num_query_elements, num_museum_elements,
                           matching_method, local_metric, threshold):

    """
    This function calculates the similarity between each image of the query set with all the museum database images,
    and then sorts out the museum images by distance in ascending order.

    :param num_museum_elements:
    :param num_query_elements:
    :param color_base: string indicating the color base in which the histogram needs to be calculated
    :param metric: string indicating which metric to use to calculate the distance
    :param dimension:
    :param query_hists: Dict containing the histograms of each of the images from the query set
    :param query_textures:
    :param query_ocrs:
    :param query_local_descriptors:
    :param museum_hists: Dict containing the histograms of each of the images from the museum database
    :param museum_textures:
    :param museum_ocrs:
    :param museum_local_descriptors
    :param matching_method: for local descriptors
    :param local metric: for local descriptors
    :param threshold: for local descriptors
    :return:
    """

    predictions = []

    for idx_query in range(num_query_elements):
        query_element_distances_list = []
        print("Calculating similarities for Query Image " + str(idx_query))
        for idx_museum in range(num_museum_elements):
            distance = 0.0
            if query_hists is not None:
                distance += calculate_hist_distance(color_base, metric, dimension, query_hists[idx_query],
                                                    museum_hists[idx_museum])

            if query_textures is not None:
                distance += calculate_hist_distance(None, 'euclidean_distance', None, query_textures[idx_query],
                                                    museum_textures[idx_museum])

            if query_ocrs is not None:
                distance += (1 - calculate_text_distance(query_ocrs[idx_query], museum_ocrs[idx_museum], 'levenshtein'))

            if query_local_descriptors is not None:
                distance = - match_descriptors(query_local_descriptors[idx_query], museum_local_descriptors[idx_museum], matching_method, local_metric, threshold)

            query_element_distances_list.append([idx_museum, distance])

        # Sort the values and remove the distances
        # d = [item[1] for item in query_element_distances_list]
        if all(item[1] == 0 for item in query_element_distances_list):
            predictions.append([-1])
        else:
            query_element_distances_list.sort(key=lambda x: x[1])
            aux_list = []
            for pair in query_element_distances_list:
                del (pair[1])
                aux_list.append(pair[0])
            predictions.append(aux_list)
    
    return predictions


def calculate_image_histogram(image, image_mask, color_base, dimension, level, x_pixel_to_split, side):
    """
    Calls the function that calculates the histogram in the specified color base for each of the pictures that are
    located in the specified path.

    :param image: string indicating the path where the images are located
    :param image_mask: mask that has to be applied to the image
    :param color_base: string indicating the color base in which the histogram needs to be calculated
    :param dimension:
    :param level:
    :param x_pixel_to_split: indicates the x pixel to split the image and mask if there are more than one painting
    :param side: indicates the side to split the image and mask if there are more than one painting
    :return: Array or matrix containing the resulting histogram
    """

    return get_image_histogram(image, image_mask, color_base, dimension, level, x_pixel_to_split, side)


def get_top_k(predictions, k, number_subimages_dic):
    """
    This function returns an array of size (n_queries x k) with the index of the images
    from the dataset that are closer to the query image.

    :param predictions: matrix containing, for each of the queries, all the museum images ordered
                        by distance in ascending order
    :param k: number of closer images to keep for each query
    :param number_subimages_dic: dictionary containing the number of subimages for each image
    :return: array of size (n_queries x k) with the top-k closer images to each of the queries
    """
    predictions_to_return = []
    if number_subimages_dic is None:
        for element in predictions:
            if element == -1:
                predictions_to_return.append(element)
            else:
                del(element[k:])
                predictions_to_return.append(element)
    else:
        predictions_idx = 0
        for idx, number_subimages in number_subimages_dic.items():
            if number_subimages == 0:
                continue
            if number_subimages == 1:
                if predictions[idx] == -1:
                    predictions_to_return.append(predictions[idx])
                else:
                    del(predictions[idx][k:])
                    predictions_to_return.append(predictions[idx])
            else:
                aux_list = []
                if predictions[predictions_idx] == -1:
                    aux_list.extend(predictions[predictions_idx + 1])
                else:
                    del (predictions[predictions_idx][k:])
                    aux_list.extend(predictions[predictions_idx])
                if predictions[predictions_idx + 1] == -1:
                    aux_list.extend(predictions[predictions_idx + 1])
                else:
                    del (predictions[predictions_idx + 1][k:])
                    aux_list.extend(predictions[predictions_idx + 1])

                predictions_to_return.append(aux_list)

            predictions_idx += 1

    return predictions_to_return


def get_mapk(GT, predictions, k):
    """
    This function returns the Mean Average Precision evaluation metric.

    :param GT: ground truth obtained from a certain pickle file
    :param predictions: matrix containing, for each of the queries, all the museum images ordered
                        by distance in ascending order
    :return: float with the score obtained
    """

    return metrics.mapk(GT, predictions, k)


def get_mask(image, masks_path, idx):
    """

    """
    return mask_creation_v2(image, masks_path, idx)


def evaluate_mask(annotation_mask, result_mask, idx):
    """
    This function calculates the Precision, Recall and F1 score by comparing the ground truth mask
    with the mask obtained with our algorithm.

    :param annotation_mask: ground truth maks
    :param result_mask: obtained masks
    :return: precision, recall and F1 score
    """

    return mask_evaluation(annotation_mask, result_mask, idx)


def detect_bounding_boxes(path, mask_set_path, method, save_masks, subpaintings, idx):
    """
    This function detects the bounding boxes of the text in all the images of a specific folder

    :param path: path of the images
    :param mask_set_path: path where the masks need to be saved
    :param method: 1 for color segmentation, 2 for morphology operations, 3 for neural network
    :param save_masks: boolean indicating if the masks need to be saved
    :param subpaintings: compute if there are one or two subpaintings in the image: False for 1, True for 2
    :param idx: int indicating the index of the image
    :return: list of bounding boxes from first image to last image. Each image contains a maximum of 2 bounding boxes.
    
        [[[first_bounding_box_of_first_image],[second_bounding_box_of_second_image]], [[first_bounding_box_of_second_image]], ...]
    
    Each bounding box has the following int values:

        [lowest_pixel_x, lowest_pixel_y, highest_pixel_x, highest_pixel_y] 
    """

    return bounding_boxes_detection(path, mask_set_path, method, save_masks, subpaintings, idx)


def evaluate_text(GT_bounding_boxes, result_bounding_boxes):
    """
    This function evaluates the accuracy of the result bounding boxes by calculating the parameter intersection over Union (IoU)
    
    :param GT_bounding_boxes: Ground Truth bounding boxes
    :param result_bounding_boxes: bounding boxes detected in the images

    :return: float with IoU parameter

    """

    return bounding_boxes_evaluation(GT_bounding_boxes, result_bounding_boxes)


def detect_paintings(query_image, mask, idx):
    """
    This function evaluates how many paintings there are in a given image, using the background mask
    of the image
    :param query_image: image to evaluate number of paintings
    :param mask: background binary mask of the image

    return: 0 if there is only one painting
            x_value_to_split: indicates the horizontal pixel where we want to split the image when there are two paintings
    """
    
    return paintings_detection(query_image, mask, idx)

    
def remove_noise(test_set_path, query_path, query_image, GT, idx, PSNR):

    # Remove noise

    image = cv2.imread(query_image)
    # denoised_image = cv2.medianBlur(image, 5)
    # denoised_image = cv2.bilateralFilter (image, 7, 100, 100);

    # Adaptative median filter
 
    denoised_image = image
    kernel_max = 13
    kernel = 1
    minimum = cv2.erode(image, np.ones((kernel,kernel), np.uint8) / kernel**2, iterations = 1)
    maximum = cv2.dilate(image, np.ones((kernel,kernel), np.uint8) / kernel**2, iterations = 1)
    median = cv2.medianBlur(image, kernel)
    
    while kernel <= kernel_max:
        kernel += 2

        minimum = cv2.erode(denoised_image, np.ones((kernel,kernel), np.uint8) / kernel**2, iterations = 1)
        maximum = cv2.dilate(denoised_image, np.ones((kernel,kernel), np.uint8) / kernel**2, iterations = 1)
        median = cv2.medianBlur(denoised_image, kernel)
        print("kernel size: ", kernel)
        for nChannel in range(image.shape[2]):
            print("Number of channel: ", nChannel)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):

                    if (minimum[i, j, nChannel] == median[i, j, nChannel]) | (median[i, j, nChannel] == maximum[i, j, nChannel]):
                        continue
                    elif (minimum[i, j, nChannel] == denoised_image[i, j, nChannel]) | (denoised_image[i, j, nChannel] == maximum[i, j, nChannel]):
                        denoised_image[i, j, nChannel] = median[i, j, nChannel]

    print("Getting denoised image " + str(idx))
    cv2.imwrite(query_path + '_denoised/' + "{0:0=5d}".format(idx) + '.jpg', denoised_image)

    # Getting original image
    if GT is not 0:
        museum_filenames = glob.glob(test_set_path + '*.jpg')
        museum_filenames.sort()
    
        museum_image = museum_filenames[int(GT[idx][0])]
        best_image = cv2.imread(museum_image)
        best_image = cv2.resize(best_image, (denoised_image.shape[1], denoised_image.shape[0]))

        # Compute PSNR
        MSE = np.mean((best_image - denoised_image) ** 2)
        if MSE == 0:
            PSNR_image = 100
        else:
            PSNR_image = 10 * np.log10((255**2) / np.sqrt(MSE))

        PSNR.append(PSNR_image)

        return PSNR
    else:
        return 0
