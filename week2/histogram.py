import cv2
import numpy as np


def calculate_1d_histogram_grayscale(image, mask):
    """
    This function calculates a 1D histogram for each of the images given

    :param image: cv2 image
    :param mask: mask to apply to the image
    :return: A dictionary where each key is the index image and the values are the 1D histograms
    """

    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], mask, [256], [0, 256])
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
        hist_image.extend(hist)

    return hist_image


def calculate_2d_histogram(image, mask, color_base):
    """

    :param image:
    :param mask:
    :param color_base:
    :return:
    """
    if color_base == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        range_hist = [0, 180, 0, 256]
        channels = [0, 1]
    elif color_base == 'LAB': #NOT WORKING AS EXPECTED
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        range_hist = [0, 256, 0, 256]
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

    if color_base == 'BGR':
        range_hist = [0, 256, 0, 256, 0, 256]
    else:
        raise Exception("Color Base is not valid")

    hist = cv2.calcHist([image], [0, 1, 2], mask, [32, 32, 32], range_hist)
    cv2.normalize(hist, hist)

    return hist.flatten()


def get_level_histograms(image, mask, color_base, dimension, num_blocks):
    """

    :param image:
    :param mask:
    :param color_base:
    :param dimension:
    :param num_blocks:
    :return:
    """

    histograms = []
    height, width = image.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))    # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = image[i:i + height_block, j:j + width_block]

            if mask is not None:
                block_mask = mask[i:i + height_block, j:j + width_block]
            else:
                block_mask = None
            #cv2.imwrite('cropped/' + str(i) + str(j) + ".png", block)
            if color_base == 'Grayscale':
                histograms.extend(calculate_1d_histogram_grayscale(block))
            elif color_base != "Grayscale" and dimension == '1D':
                histograms.extend(calculate_1d_histogram_color(block, block_mask, color_base))
            elif color_base != "Grayscale" and dimension == '2D':
                histograms.extend(calculate_2d_histogram(block, block_mask, color_base))
            elif color_base != "Grayscale" and dimension == '3D':
                histograms.extend(calculate_3d_histogram(block, block_mask, color_base))

    return histograms


def get_image_histogram(image, mask, color_base, dimension, level, x_pixel_to_split, side):
    """
    This class calls the functions that calculate the histograms, depending if it is a 1D or 3D histogram

    :param image: String indicating the path where the images are located
    :param mask: mask to apply to the image. If mask == None, no mask is applied
    :param color_base: String that indicates in which color base the histogram has to be calculated
    :param dimension:
    :param level:
    :param x_pixel_to_split: indicates the x pixel to split the image and mask if there are more than one painting
    :param side: indicates the side to split the image and mask if there are more than one painting
    :return: a dictionary where the keys are the index of the images and the values are the histograms
    """

    image = cv2.imread(image)

    if (x_pixel_to_split != None):
        if (side == "left"):
            image = image[1:image.shape[0], 1:int(x_pixel_to_split)]
            mask = mask[1:mask.shape[0], 1:int(x_pixel_to_split)]

        elif (side == "right"):
            image = image[1:image.shape[0], int(x_pixel_to_split):image.shape[1]]
            mask = mask[1:mask.shape[0], int(x_pixel_to_split):mask.shape[1]]

    if level == 1:
        if color_base == 'Grayscale':
            return calculate_1d_histogram_grayscale(image, mask)
        elif color_base != "Grayscale" and dimension == '1D':
            return calculate_1d_histogram_color(image, mask, color_base)
        elif color_base != "Grayscale" and dimension == '2D':
            return calculate_2d_histogram(image, mask, color_base)
        elif color_base != "Grayscale" and dimension == '3D':
            return calculate_3d_histogram(image, mask, color_base)
    elif level == 2 or level == 3:
        histogram = []

        for level in range(level + 1):
            number_of_blocks = 2**level
            histogram.extend(get_level_histograms(image, mask, color_base, dimension, number_of_blocks))

        return histogram
    else:
        raise Exception("The selected level is not correct")

