import cv2
from skimage import feature
import numpy as np


def LBP_descriptor(image, num_blocks):
    """
    This function calculates the LBP descriptor for a given image.

    :param image:
    :param num_blocks:
    :return:
    """

    descriptor = []
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = grayscale_image[i:i + height_block, j:j + width_block]
            block_lbp = np.float32(feature.local_binary_pattern(block, 8, 2, method='default'))
            hist = cv2.calcHist([block_lbp], [0], None, [16], [0, 255])
            cv2.normalize(hist, hist)
            descriptor.extend(hist.tolist())

    return descriptor


def DCT_descriptor(image, num_blocks):
    """
    :param image:
    :param num_blocks:
    :return:
    """

    descriptor = []
    number_coefficients = 100
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = grayscale_image[i:i + height_block, j:j + width_block]

            # Step 1: Calculate the DCT
            block_dct = cv2.dct(np.float32(block)/255.0)
            cv2.imshow("Image", block_dct)
            cv2.waitKey(0)

            # Step 2: Zig-Zag scan
            zig_zag_scan = np.concatenate([np.diagonal(block_dct[::-1, :], i)[::(2*(i % 2)-1)]
                                           for i in range(1-block_dct.shape[0], block_dct.shape[0])])

            # Step 3: Keep first N coefficients
            descriptor.extend(zig_zag_scan.tolist()[:number_coefficients])

    return descriptor


def HOG_descriptor(image):
    """
    Computes the HOG (Histogram of Oriented Gradients) of the given image. By default, HOG uses 9 levels.
    :param image: image to which the HOG will be calculated
    :return: the HOG of the given image
    """
    """TODO"""
    #hog = cv2.HOGDescriptor()
    #return hog.compute(image)
    pass


def get_image_descriptor(image, descriptor, descriptor_level):
    """

    :param image:
    :param descriptor:
    :param descriptor_level:
    :param mask:
    :return:
    """

    # Check if the descriptor level is big enough
    if descriptor_level < 3:
        descriptor_level = 3

    number_of_blocks = 2**(descriptor_level - 1)
    im = cv2.imread(image)

    if descriptor == "LBP":
        return LBP_descriptor(im, number_of_blocks)
    elif descriptor == "LBP_multiscale":
        return LBP_descriptor(im, number_of_blocks)
    elif descriptor == "DCT":
        return DCT_descriptor(im, number_of_blocks)
    elif descriptor == "HOG":
        return HOG_descriptor(im)
    else:
        raise Exception("Image descriptor is not valid")
