import cv2
import numpy as np

def calculate_1d_histogram_grayscale(image):
    """
    This function calculates a 1D histogram for each of the images given

    :param image: cv2 image
    :return: A dictionary where each key is the index image and the values are the 1D histograms
    """

    image = cv2.imread(image)
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)

    return hist


def calculate_1d_histogram_color(image, mask, color_base):
    """
    This function calculates 3x1D histogram for each of the images given. Several color bases are supported

    :param image: cv2 image
    :param mask: mask to apply to the image. If mask == None, no mask is applied
    :param color_base: String that indicates in which color base the histogram has to be calculated
    :return: A dictionary where each key is the index image and the values are the 3D histograms
    """

    hist_image = []
    image = cv2.imread(image)

    if color_base == 'BGR':
        color = ('b', 'g', 'r')
    elif color_base == 'LAB':
        color = ('L', 'b', 'a')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_base == 'YCrCb':
        color = ('Y', 'Cr', 'b')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_base == 'HSV':
        color = ('H', 'S', 'V')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise Exception("Color Base is not valid")

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
        cv2.normalize(hist, hist)
        hist_image.append(hist)

    return hist_image


def calculate_2d_histogram(image, mask, color_base):
    """

    :param image:
    :param mask:
    :param color_base:
    :return:
    """

    image = cv2.imread(image)

    if color_base == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        range_hist = [0, 180, 0, 256]
        channels = [0, 1]
    elif color_base == 'LAB': #NOT WORKING AS EXPECTED
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        range_hist = [-127, 127, -127, 127]
        channels = [1, 2]
    elif color_base == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = [1, 2]
        range_hist = [0, 256, 0, 256]
    else:
        raise Exception("Color Base Not Valid")

    hist = cv2.calcHist([image], channels, mask, [128, 128], range_hist)
    cv2.normalize(hist, hist)

    return hist.flatten()


def calculate_3d_histogram(image, mask, color_base):
    """

    :param image:
    :param mask:
    :param color_base:
    :return:
    """

    image = cv2.imread(image)

    if color_base == 'BGR':
        range_hist = [0, 256, 0, 256, 0, 256]
    else:
        raise Exception("Color Base is not valid")

    hist = cv2.calcHist([image], [0, 1, 2], mask, [128, 128, 128], range_hist)
    cv2.normalize(hist, hist)

    return hist.flatten()


def get_image_histogram(image, mask, color_base, dimension):
    """
    This class calls the functions that calculate the histograms, depending if it is a 1D or 3D histogram

    :param path: String indicating the path where the images are located
    :param mask: mask to apply to the image. If mask == None, no mask is applied
    :param color_base: String that indicates in which color base the histogram has to be calculated
    :param dimension:
    :return: a dictionary where the keys are the index of the images and the values are the histograms
    """

    if color_base == 'Grayscale':
        return calculate_1d_histogram_grayscale(image)
    elif color_base != "Grayscale" and dimension == '1D':
        return calculate_1d_histogram_color(image, mask, color_base)
    elif color_base != "Grayscale" and dimension == '2D':
        return calculate_2d_histogram(image, mask, color_base)
    elif color_base != "Grayscale" and dimension == '3D':
        return calculate_3d_histogram(image, mask, color_base)
