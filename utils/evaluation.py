import pickle
from utils import distances_metrics, histogram, mask
import numpy as np
import ml_metrics as metrics


def get_ground_truth(path):
    """
    Returns the ground truth stored in a pickle file as a list of lists

    :param path: path where the ground truth is located
    :return: ground truth as a list of lists of stringss
    """

    pkl_file = open(path, 'rb')
    return pickle.load(pkl_file, fix_imports=True, encoding='ASCII', errors='strict')


def calculate_hist_distance(colorBase, metric, histA, histB):
    """
    This function calculates the distance between 2 histograms in the chosen metric.

    :param colorBase: string indicating the color base in which the histogram has been calculated
    :param metric: string indicating which metric to use to calculate the distance
    :param histA: 1x256 array or 1x768 array containing histogram 1
    :param histB: 1x256 array or 1x768 array containing histogram 2
    :return: float indicating the distance between the two histograms
    """

    distance = 0
    if colorBase == '1D':
        distance = distances_metrics.compute_distance(histA, histB, metric)
    else:
        bins = 256
        aux_A = np.reshape(histA, (3, bins))
        aux_B = np.reshape(histB, (3, bins))
        for i in range(3):
            distance += distances_metrics.compute_distance(aux_A[i], aux_B[i], metric)

    return distance


def calculate_similarities(colorBase, metric, QS_Histograms, DB_Histograms):
    """
    This function calculates the similarity between each image of the query set with all the museum database images,
    and then sorts out the museum images by distance in ascending order.

    :param colorBase: string indicating the color base in which the histogram needs to be calculated
    :param metric: string indicating which metric to use to calculate the distance
    :param QS_Histograms: List containing the histograms of each of the images from the query set
    :param DB_Histograms: List containing the histograms of each of the images from the museum database
    :return:
    """
    predictions = []

    for query_hist in QS_Histograms.values():
        query_element_distances_list = []
        idx_museum = 0
        for museum_hist in DB_Histograms.values():
            distance = calculate_hist_distance(colorBase, metric, query_hist, museum_hist)
            query_element_distances_list.append([idx_museum, distance])
            idx_museum += 1

        # Sort the values and remove the distances
        query_element_distances_list.sort(key=lambda x: x[1])
        aux_list = []
        for pair in query_element_distances_list:
            del(pair[1])
            aux_list.append(pair[0])

        predictions.append(aux_list)

    return predictions


def calculate_image_histograms(path, colorBase):
    """
    Calls the function that calculates the histogram in the specified color base for each of the pictures that are
    located in the specified path.

    :param path: string indicating the path where the images are located
    :param colorBase: string indicating the color base in which the histogram needs to be calculated
    :return: 1x256 array or 1x768 array containing the histogram
    """

    return histogram.get_histograms(path, colorBase)


def get_top_K(predictions, k):
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

def get_mask(path):
    """
    This function calls the function that calculates a mask based on color segmentation and create the mask
    image (horizontal_pixels x vertical_pixels) inside the same folder as the image and with following name:
    
        MASK: path + number_of_image + _mask.png 
    
    It also creates the images with the background remove in the same folder with name

        IMAGE_WITH_MASK: path + number_of_image + _image_with_mask.png

    The function returns true if it has finished without any error

    :param path: string indicating the path where the images are located
    :return: true
    """
    print("im here")
    return mask.maskCreation(path)
