import cv2
import numpy as np
from scipy.spatial import distance


def _distance(u, v):
    """
    Compare the image descriptor vectors based on a distance metric.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between the image descriptor vectors.
    """

    pass


def correlation(u, v):
    """
    Compare the histograms based on correlation.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between the histograms.
    """

    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_CORREL)


def chi_square(u, v):
    """
    Compare the histograms based on the Chi-Square test.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between the histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_CHISQR)


def bhattacharya_distance(u, v):
    """
    Compare histograms based on the Bhattacharyya distance.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between the histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_BHATTACHARYYA)


def intersection(u, v):
    """
    Compare histograms based on their intersection.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between the histograms.
    """
    return 1 - cv2.compareHist(u, v, cv2.HISTCMP_INTERSECT)


def kl_divergence(u, v):
    """
    Compare histograms based on the Kullback-Leibler divergence.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between histograms.
    """

    return cv2.compareHist(u, v, cv2.HISTCMP_KL_DIV)


def euclidean_distance(u, v):
    """
    Compare descriptor vectors based on euclidian distance.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between descriptor vectors.
    """

    return distance.euclidean(u, v)


def l1_distance(u, v):
    """
    Compare descriptor vectors based on L1 distance.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between descriptor vectors.
    """

    return distance.minkowski(u, v, 1)



def cosine_distance(u, v):
    """
    Compare the image descriptor vectors based on the cosine similarity between them.
    Args:
        u (ndarray): 1D array of type np.float32 containing image descriptors.
        v (ndarray): 1D array of type np.float32 containing image descriptors.
    Returns:
        float: computed distance between descriptor vectors.
    """
    return distance.cosine(u, v)


def compute_distance(u, v, metric):
    u = np.array(u)
    v = np.array(v)

    func = {
        'euclidean_distance': euclidean_distance,
        'l1_distance': l1_distance,
        'cosine_distance': cosine_distance,
        'correlation': correlation,                         #cv2
        'chi_square': chi_square,                           #cv2
        'intersection': intersection,                       #cv2
        'bhattacharya_distance': bhattacharya_distance,     #cv2
        'kl_divergence': kl_divergence
    }
    return func[metric](u, v)
