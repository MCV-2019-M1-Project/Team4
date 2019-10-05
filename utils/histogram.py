import numpy as np
import cv2
import glob


def calculate_1d_histogram(filenames):
    """
    This function calculates a 1D histogram for each of the images given

    :param filenames: list of images
    :return: A dictionary where each key is the index image and the values are the 1D histograms
    """

    hist_1d_global = {}  # Declaration of the return dic: key is the index of the image and value is the histogram

    idx = 0
    for image in filenames:  # Loop for each image

        # Compute the histogram for every image
        im = cv2.imread(image)
        hist, bins = np.histogram(im.ravel(), 256, [0, 256])
        hist = [float(i)/max(hist) for i in hist]

        hist_1d_global[idx] = hist
        idx += 1

    return hist_1d_global


def calculate_3d_histogram(filenames, color_base):
    """
    This function calculates a 3D histogram for each of the images given. Several color bases are supported

    :param filenames: list of images
    :param color_base: String that indicates in which color base the histogram has to be calculated
    :return: A dictionary where each key is the index image and the values are the 3D histograms
    """

    hist_image_set = {}
    idx = 0

    for image in filenames:
        hist_image = []
        im = cv2.imread(image)

        if color_base == 'BGR':
            color = ('b','g','r')
        elif color_base == 'LAB':
            color = ('L','b','a')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        elif color_base == 'YCrCb':
            color = ('Y','Cr','b')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        elif color_base == 'HSV':
            color = ('H','S','V')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        else:
            raise Exception("Color Base is not valid")

        for i, col in enumerate(color):
            hist = cv2.calcHist([im], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            hist_image.append(hist)

        hist_image_set[idx] = hist_image
        idx += 1

    return hist_image_set


def get_histograms(path, color_base):
    """
    This class calls the functions that calculate the histograms, depending if it is a 1D or 3D histogram

    :param path: String indicating the path where the images are located
    :param color_base: String that indicates in which color base the histogram has to be calculated

    :return: a dictionary where the keys are the index of the images and the values are the histograms
    """

    # read images in dataset
    filenames = glob.glob(path)
    filenames.sort()

    if color_base == '1D':
        return calculate_1d_histogram(filenames)
    else:
        return calculate_3d_histogram(filenames, color_base)
