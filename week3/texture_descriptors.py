import cv2
from skimage import feature
import numpy as np


def LBP_descriptor(image, num_blocks, mask):
    """
    This function calculates the LBP descriptor for a given image.

    :param image: image used to calculate the LBP function
    :param num_blocks: number of blocks in which both the height and the width will be divided into
    :param mask: binary mask that will be applied to the image
    :return: the LBP feature array
    """

    descriptor = []
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    height, width = grayscale_image.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = grayscale_image[i:i + height_block, j:j + width_block]

            if mask is not None:
                block_mask = mask[i:i + height_block, j:j + width_block]
            else:
                block_mask = None

            block_lbp = np.float32(feature.local_binary_pattern(block, 8, 2, method='default'))

            if mask is not None:
                mask = mask[i:i + height_block, j:j + width_block]

            hist = cv2.calcHist([block_lbp], [0], block_mask, [16], [0, 255])
            cv2.normalize(hist, hist)
            descriptor.extend(hist)

    return descriptor


def DCT_descriptor(image, num_blocks, mask):
    """
    This function calculates the DCT texture descriptor for the given image.
    :param image: image used to calculate the DCT function
    :param num_blocks: number of blocks in which both the height and the width will be divided into
    :param mask: binary mask that will be applied to the image
    :return: the DCT feature array
    """

    descriptor = []
    number_coefficients = 100
    resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    if mask is not None:
        resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
        resized_image = cv2.bitwise_and(resized_image, resized_image, mask=resized_mask)

    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape[:2]
    height_block = int(np.ceil(height / num_blocks))  # Number of height pixels for sub-image
    width_block = int(np.ceil(width / num_blocks))  # Number of width pixels for sub-image

    for i in range(0, height, height_block):
        for j in range(0, width, width_block):
            block = grayscale_image[i:i + height_block, j:j + width_block]

            # Step 1: Calculate the DCT
            block_dct = cv2.dct(np.float32(block)/255.0)

            # Step 2: Zig-Zag scan
            zig_zag_scan = np.concatenate([np.diagonal(block_dct[::-1, :], i)[::(2*(i % 2)-1)]
                                           for i in range(1-block_dct.shape[0], block_dct.shape[0])])

            # Step 3: Keep first N coefficients
            descriptor.extend(zig_zag_scan[:number_coefficients])

    return descriptor


def HOG_descriptor(image, mask):
    """
    Computes the HOG (Histogram of Oriented Gradients) of the given image.
    :param image: image to which the HOG will be calculated
    :param mask: binary mask that will be applied to the image
    :return: array with the image features
    """
    grayscale = True
    multichannel = True

    resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    if grayscale:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        multichannel = False

    if mask is not None:
        resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
        resized_image = cv2.bitwise_and(resized_image, resized_image, mask=resized_mask)

    return feature.hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                       block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True,
                       multichannel=multichannel)


def get_image_texture_descriptor(image, descriptor, descriptor_level, mask):
    """
    This functions returns a feature array for a given image. Supported methods to extract the feture vector are LBP,
    DCT and HOG
    :param image: image to which the HOG will be calculated
    :param descriptor: LBP, DCT or HOG descriptors
    :param descriptor_level: integer indicating the descriptor level. It is used to calculate the number of blocks in
    which the width and the height will be divided into (LBP and DCT descriptors only)
    :param mask: binary mask that will be applied to the image
    :return: feature array of the given image
    """

    # Check if the descriptor level is big enough
    if descriptor_level < 3:
        descriptor_level = 3

    number_of_blocks = 2**(descriptor_level - 1)
    im = cv2.imread(image)

    if descriptor == "LBP":
        return LBP_descriptor(im, number_of_blocks, mask)
    elif descriptor == "DCT":
        return DCT_descriptor(im, number_of_blocks, mask)
    elif descriptor == "HOG":
        return HOG_descriptor(im, mask)
    else:
        raise Exception("Image descriptor is not valid")
