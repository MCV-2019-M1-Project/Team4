from __future__ import division

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def sift_descriptors(image, mask):
    """
    Extract descriptors from image using the SIFT method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): list of cv2.KeyPoint objects.
    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 128)
            containing local descriptors for the keypoints.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grayscale_image, mask=mask)

    # image = cv2.drawKeypoints(grayscale_image,keypoints,image)

    # cv2.imshow(image, image)

    return descriptors


def surf_descriptors(image, mask):
    """
    Extract descriptors from image using the SURF method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): list of cv2.KeyPoint objects.
    Returns:
        descriptors (ndarray): 2D array of type np.float32 and shape (#keypoints x 64)
            containing local descriptors for the keypoints.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(grayscale_image, mask)
    return descriptors


def root_sift_descriptors(image, mask, eps=1e-7):
    """
    Extract descriptors from image using the RootSIFT method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): list of cv2.KeyPoint objects.
    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.
    """

    descs = sift_descriptors(image, mask)
    if descs is not None:
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
    
    return descs


def orb_descriptors(image, mask):
    """
    Extract descriptors from keypoints using the ORB method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): list of cv2.KeyPoint objects.
    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.
    """
    
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(WTA_K=4)
    keypoints, descriptors = orb.detectAndCompute(grayscale_image, mask)

    return descriptors


def daisy_descriptors(image, mask):
    """
    Extract descriptors from keypoints using the Daisy method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): Not used
    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(grayscale_image ,mask)

    daisy = cv2.xfeatures2d.DAISY_create()
    _, descriptors = daisy.compute(grayscale_image, keypoints)
    
    return descriptors


def brisk_descriptors(image, mask):
    """
    Extract descriptors from keypoints using the Daisy method.
    Args:
        image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
        mask:
        keypoints (list): Not used
    Returns:
        descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.
    """

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.resize(grayscale_image, (256, 256), interpolation=cv2.INTER_AREA)

    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)

    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(grayscale_image, mask)

    return descriptors


# def lbp(image, keypoints):
#     """
#     Extract descriptors from keypoints using the Local Binary Pattern method.
#     Args:
#         image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
#         keypoints (list): Not used
#     Returns:
#         descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.
#     """

#     result = []
#     for kp in keypoints:
#         img = image[round(kp.pt[1] - kp.size/2):round(kp.pt[1] + kp.size/2),
#         round(kp.pt[0] - kp.size/2):round(kp.pt[0] + kp.size/2)]

#         numPoints = 30
#         radius = 2
#         eps = 1e-7

#         lbp = local_binary_pattern(img, numPoints, radius, method="uniform")
#         (hist, _) = np.histogram(lbp.ravel(),
#                                  bins=np.arange(0, numPoints + 3),
#                                  range=(0, numPoints + 2))

#         # normalize the histogram
#         hist = hist.astype("float")
#         hist /= (hist.sum() + eps)

#         result.append(np.array(hist, dtype=np.float32))

#     # return the histogram of Local Binary Patterns
#     return result


# def hog_descriptor(image, keypoints):
#     """
#     Extract descriptors from keypoints using the Histogram of Gradients method.
#     Args:
#         image (ndarray): (H x W) 2D array of type np.uint8 containing a grayscale image.
#         keypoints (list): Not used
#     Returns:
#         descriptors (ndarray): 2D array of type np.float32 containing local descriptors for the keypoints.
#     """
#     hog = cv2.HOGDescriptor()
#     result = []
#     for kp in keypoints:
#         descriptor = hog.compute(image, locations=[kp.pt])
#         if descriptor is None:
#             descriptor = []
#         else:
#             descriptor = descriptor.ravel()
#         result.append(np.array(descriptor, dtype=np.float32))
#     return result


def extract_local_descriptors(image, mask, method, x_pixel_to_split, side):

    if type(image) == str:
        image = cv2.imread(image)

    if x_pixel_to_split is not None:
        if side == "left":
            image = image[1:image.shape[0], 1:int(x_pixel_to_split)]
            mask = mask[1:mask.shape[0], 1:int(x_pixel_to_split)]

        elif side == "right":
            image = image[1:image.shape[0], int(x_pixel_to_split):image.shape[1]]
            mask = mask[1:mask.shape[0], int(x_pixel_to_split):mask.shape[1]]

    descriptors = {
        'sift': sift_descriptors,
        'surf': surf_descriptors,
        'root_sift': root_sift_descriptors,
        'orb': orb_descriptors,
        'fast-daisy': daisy_descriptors,
        'brisk': brisk_descriptors,
        # 'hog': hog_descriptor,
        # 'lbp': lbp
    }
    return descriptors[method](image, mask)
