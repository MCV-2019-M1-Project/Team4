import numpy as np
import cv2


def mask_creation(image, mask_path, image_index):
    """
    Method to create a mask for each of the images located in a given path

    :param image:
    :param mask_path: String indicating the path where the masks will be saved
    :param image_index:
    :return: (True, mask)
    """
    # convert image to hsv color space
    image = cv2.imread(image)

    im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(im_hsv)

    # compute the mean value of hue, saturation and value for the border of the image
    hue_mean_border = (np.mean(h[0, :]) + np.mean(h[:, 0]) + np.mean(h[-1, :]) + np.mean(h[:, -1]))/4
    saturation_mean_border = (np.mean(s[0, :]) + np.mean(s[:, 0]) + np.mean(s[-1, :]) + np.mean(s[:, -1]))/4
    value_mean_border = (np.mean(v[0, :]) + np.mean(v[:, 0]) + np.mean(v[-1, :]) + np.mean(v[:, -1]))/4

    # compute lower and upper limits for the mask
    # we need to find the good limits to segment the background by color
    lower_hue = (hue_mean_border - 20)
    upper_hue = (hue_mean_border + 20)
    lower_saturation = (saturation_mean_border - 30)
    upper_saturation = (saturation_mean_border + 30)
    lower_value = (value_mean_border - 150)
    upper_value = (value_mean_border + 150)

    lower_limit = np.array([lower_hue, lower_saturation, lower_value])
    upper_limit = np.array([upper_hue, upper_saturation, upper_value])

    # create mask
    mask = cv2.inRange(im_hsv, lower_limit, upper_limit)
    mask = cv2.bitwise_not(mask)

    # apply mask to find contours
    mask = np.uint8(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create new mask with the contours found
    new_mask = cv2.fillPoly(mask, contours, [255, 255, 255])

    # ------- image_with_mask = cv2.bitwise_and(im, im, mask = new_mask) ----------
    # print(im)

    # save mask image inside the same folder as the image
    cv2.imwrite(mask_path + "a" + str(image_index).zfill(2) + "_mask.png", new_mask)

    # save image with mask applied in same folder
    # cv2.imwrite(path + str(idx) + "_image_with_mask.png", image_with_mask)

    return new_mask
