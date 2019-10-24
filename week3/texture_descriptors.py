import cv2
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
from week3.evaluation import *


def LBP_descriptor(image, num_blocks, multiscale):
    """
    :param image:
    :param num_blocks:
    :return:
    """

    descriptor = []
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = grayscale_image.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    """for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = grayscale_image[i:i + height_block, j:j + width_block]

            if multiscale:
                radius = 3
                num_points = 3*radius
                block_lbp = local_binary_pattern(block, num_points, radius, method='uniform')
                print(block_lbp)
            else:
                block_lbp = local_binary_pattern(block, 9, 1, method='uniform')
                print(block_lbp)

            # Calculate the histogram
            block_histogram = itemfreq(block_lbp.ravel())
            # Normalize the histogram
            block_histogram = block_histogram[:, 1] / sum(block_histogram[:, 1])

            print(descriptor)
            descriptor.extend(block_histogram)"""

    return descriptor


def DCT_descriptor(image, num_blocks):
    """
    :param image:
    :param num_blocks:
    :return:
    """
    descriptor = []
    return descriptor


def HOG_descriptor(image):
    """
    Computes the HOG (Histogram of Oriented Gradients) of the given image. By default, HOG uses 9 levels.
    :param image: image to which the HOG will be calculated
    :return: the HOG of the given image
    """

    hog = cv2.HOGDescriptor()
    return hog.compute(image)


def get_image_descriptor(image, descriptor, descriptor_level):
    """

    :param image:
    :param descriptor:
    :param descriptor_level:
    :return:
    """

    # Check if the descriptor level is big enough
    if descriptor_level < 3:
        descriptor_level = 3

    number_of_blocks = 2**(descriptor_level - 1)
    im = cv2.imread(image)

    if descriptor == "LBP":
        return LBP_descriptor(im, number_of_blocks, False)
    elif descriptor == "LBP_multiscale":
        return LBP_descriptor(im, number_of_blocks, True)
    elif descriptor == "DCT":
        return DCT_descriptor(im, number_of_blocks)
    elif descriptor == "HOG":
        return HOG_descriptor(im)
    else:
        raise Exception("Image descriptor is not valid")
