import pickle

from week2 import distances_metrics, histogram, mask, text  
import ml_metrics as metrics
import numpy as np


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
            distance += distances_metrics.compute_distance(aux_a[i], aux_b[i], metric)
    else:
        distance = distances_metrics.compute_distance(hist_a, hist_b, metric)

    return distance


def calculate_similarities(color_base, metric, dimension, QS_Histograms, DB_Histograms):
    """
    This function calculates the similarity between each image of the query set with all the museum database images,
    and then sorts out the museum images by distance in ascending order.

    :param color_base: string indicating the color base in which the histogram needs to be calculated
    :param metric: string indicating which metric to use to calculate the distance
    :param QS_Histograms: Dict containing the histograms of each of the images from the query set
    :param DB_Histograms: Dict containing the histograms of each of the images from the museum database
    :return:
    """
    predictions = []

    idx_query = 0
    for query_hist in QS_Histograms.values():
        query_element_distances_list = []
        idx_museum = 0
        print("Calculating similarities for Query Image " + str(idx_query))
        for museum_hist in DB_Histograms.values():
            distance = calculate_hist_distance(color_base, metric, dimension, query_hist, museum_hist)
            query_element_distances_list.append([idx_museum, distance])
            idx_museum += 1
        idx_query += 1

        # Sort the values and remove the distances
        query_element_distances_list.sort(key=lambda x: x[1])
        aux_list = []
        for pair in query_element_distances_list:
            del(pair[1])
            aux_list.append(pair[0])

        predictions.append(aux_list)

    return predictions


def calculate_image_histogram(image, image_mask, color_base, dimension, level):
    """
    Calls the function that calculates the histogram in the specified color base for each of the pictures that are
    located in the specified path.

    :param image: string indicating the path where the images are located
    :param image_mask: mask that has to be applied to the image
    :param color_base: string indicating the color base in which the histogram needs to be calculated
    :param dimension:
    :param level:
    :return: Array or matrix containing the resulting histogram
    """
    return histogram.get_image_histogram(image, image_mask, color_base, dimension, level)


def get_top_k(predictions, k):
    """
    This function returns an array of size (n_queries x k) with the index of the images
    from the dataset that are closer to the query image.

    :param predictions: matrix containing, for each of the queries, all the museum images ordered
                        by distance in ascending order
    :param k: number of closer images to keep for each query
    :return: array of size (n_queries x k) with the top-k closer images to each of the queries
    """

    for element in predictions:
        del(element[k:])

    return predictions


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
    return mask.mask_creation(image, masks_path, idx)

def evaluate_mask(annotation_mask, result_mask):
    """
    This function calculates the Precision, Recall and F1 score by comparing the ground truth mask
    with the mask obtained with our algorithm.

    :param annotation_mask: ground truth maks
    :param result_mask: obtained masks
    :return: precision, recall and F1 score
    """

    return mask.mask_evaluation(annotation_mask, result_mask)


def detect_bounding_boxes(path):
    """
    This function detects the bounding boxes of the text in all the images of a specific folder

    :param path: path of the images
    :return: list of bounding boxes from first image to last image. Each image contains a maximum of 2 bounding boxes.
    
        [[[first_bounding_box_of_first_image],[second_bounding_box_of_second_image]], [[first_bounding_box_of_second_image]], ...]
    
    Each bounding box has the following int values:

        [lowest_pixel_x, lowest_pixel_y, highest_pixel_x, highest_pixel_y] 
    """

    return text.bounding_boxes_detection(path)


def evaluate_text(GT_bounding_boxes, result_bounding_boxes):
    """
    This function evaluates the accuracy of the result bounding boxes by calculating the parameter intersection over Union (IoU)
    
    :param GT_bounding_boxes: Ground Truth bounding boxes
    :param result_bounding_boxes: bounding boxes detected in the images

    :return: float with IoU parameter

    """

    return text.bounding_boxes_evaluation(GT_bounding_boxes, result_bounding_boxes)