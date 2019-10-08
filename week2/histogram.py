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
        im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_base == 'YCrCb':
        color = ('Y', 'Cr', 'b')
        im = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_base == 'HSV':
        color = ('H', 'S', 'V')
        im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise Exception("Color Base is not valid")

    for i, col in enumerate(color):
        hist = cv2.calcHist([im], [i], mask, [256], [0, 256])
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
        im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        range_hist = [0, 180, 0, 256]
        channels = [0, 1]
        bins = [180, 256]
    elif color_base == 'LAB':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        range_hist = [-100, 100, -100, 100]
        channels = [1, 2]
        bins = [200, 200]
    elif color_base == 'YCrCb':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = [1, 2]
        range_hist = [0, 256, 0, 256]
        bins = [256, 256]
    else:
        raise Exception("Color Base Not Valid")

    hist = cv2.calcHist([im], channels, mask, bins, range_hist)
    cv2.normalize(hist, hist)
    return hist


def calculate_3d_histogram(image, mask, color_base):
    """

    :param image:
    :param mask:
    :param color_base:
    :return:
    """

    image = cv2.imread(image)

    if color_base == 'BGR':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bins = [256, 256, 256]
        range_hist = [0, 256, 0, 256, 0, 256]
    elif color_base == 'LAB':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        bins = [256, 256, 256]
        range_hist = [0, 100, -100, 100, -100, 100]
    elif color_base == 'HSV':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bins = [180, 256, 256]
        range_hist = [0, 180, 0, 256, 0, 256]
    elif color_base == 'YCrCb':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        bins = [256, 256, 256]
        range_hist =[0, 256, 0, 256, 0, 256]
    else:
        raise Exception("Color Base is not valid")

    hist = cv2.calcHist([im], [0, 1, 2], mask, bins, range_hist)
    cv2.normalize(hist, hist)
    return hist


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
