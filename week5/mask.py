import numpy as np
import cv2
from skimage import feature, img_as_ubyte

# Temp imports
import glob


def MSE(a,b,axis):
    """
    This function computes the MSE between a and b along the specified axis.
    Parameters
    ----------
    a : Numpy array.
    b : Numpy array.
    Returns
    -------
    Numpy array containing the MSE computation between a and b along the specified axis.
    """
    return ((a-b)**2).mean(axis=axis)


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
    lower_hue = (hue_mean_border - 40)
    upper_hue = (hue_mean_border + 40)
    lower_saturation = (saturation_mean_border - 20)
    upper_saturation = (saturation_mean_border + 20)
    lower_value = (value_mean_border - 200)
    upper_value = (value_mean_border + 200)

    lower_limit = np.array([lower_hue, lower_saturation, lower_value])
    upper_limit = np.array([upper_hue, upper_saturation, upper_value])

    # create mask
    mask = cv2.inRange(im_hsv, lower_limit, upper_limit)
    mask = cv2.bitwise_not(mask)

    # resize masks
    n_mask, m_mask = mask.shape[0], mask.shape[1]
    mask = cv2.resize(mask, (1000, 1000))

    # apply mask to find contours
    mask = np.uint8(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create new mask with the contours found
    mask_contours = cv2.fillPoly(mask, contours, [255, 255, 255])

    # Apply morphological filter to clean
    kernel = np.ones((9, 9), np.float32)/25
    mask_erode = cv2.morphologyEx(mask_contours, cv2.MORPH_ERODE, kernel, iterations = 1)
    mask_dilate = cv2.morphologyEx(mask_erode, cv2.MORPH_DILATE, kernel, iterations = 1)

    # resize masks to original size
    new_mask = cv2.resize(mask_dilate, (m_mask, n_mask))

    # save mask image inside the same folder as the image
    cv2.imwrite(mask_path + str(image_index).zfill(2) + "_mask.png", new_mask)

    return new_mask


def mask_evaluation(annotation_mask, result_mask, idx):
    """
    This function calculates the Precision, Recall and F1 score by comparing the ground truth mask
    with the mask obtained with our algorithm.

    :param annotation_mask: ground truth maks
    :param result_mask: obtained masks
    :return: precision, recall and F1 score
    """

    true_positive = np.sum(np.logical_and(annotation_mask == 255, result_mask == 255))
    false_positive = np.sum(np.logical_and(result_mask == 255, annotation_mask != result_mask))
    false_negative = np.sum(np.logical_and(annotation_mask == 255, annotation_mask != result_mask))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_measure = 2 * ((precision * recall) / (precision + recall))

    return recall, precision, f1_measure


def paintings_detection(query_image, mask):
    """
    This function evaluates how many paintings there are in a given image, using the background mask
    of the image
    :param query_image: image to evaluate number of paintings
    :param mask: background binary mask of the image

    return: 0 if there is only one painting
            x_value_to_split: indicates the horizontal pixel where we want to split the image when there are two paintings
    """

    image = cv2.imread(query_image)

    image_width = mask.shape[0]
    image_height = mask.shape[1]
    x_box_1, y_box_1, w_box_1, h_box_1, x_box_2, y_box_2, w_box_2, h_box_2 = 0, 0, 0, 0, 0, 0, 0, 0,

    contours, _ = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if (w > 0.15 * image_width) & (h > 0.15 * image_height) & (w < 0.98 * image_width) & (x_box_1 == 0):
            x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
        elif (w > 0.15 * image_width) & (h > 0.15 * image_height) & (w < 0.98 * image_width) & (x_box_1 != 0):
            x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h

    if x_box_2 == 0:
        x_value_to_split = 0
    else:
        x_value_to_split = (x_box_1 + w_box_1/2 + x_box_2 + w_box_2/2) / 2


    return(x_value_to_split)


def mask_creation_v2(image_path, path, idx):
    """

    :param image:
    :param idx:
    :return:
    """

    image = cv2.imread(image_path)
    image = cv2.medianBlur(image, 9)

    canny_tresh1 = 10
    canny_tresh2 = 30
    canny_image = cv2.Canny(image, canny_tresh1, canny_tresh2)
    canny_image = cv2.dilate(canny_image, None, iterations=7)

    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_info = []
    
    for contour in contours:
        contours_info.append((
            contour,
            cv2.contourArea(contour),
        ))
    contours_info = sorted(contours_info, key=lambda c: c[1], reverse=True)
    biggest_contour = contours_info[0]

    mask = np.zeros_like(canny_image)

    for contour in contours_info:
        if contour[1] > biggest_contour[1]*0.1:
            cv2.fillConvexPoly(mask, contour[0], [255, 255, 255])

    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=9)

    cv2.imwrite('masks/' + str(idx).zfill(2) + "_mask.png", mask)
    return mask


if __name__ == '__main__':
    query_filenames = glob.glob("images/qsd1_w4_denoised/*.jpg")
    query_filenames.sort()
    masks = {}
    print(query_filenames)
    idx = 0

    for query in query_filenames:
        if idx == 12:
            #pass
            masks[idx] = mask_creation_v2(query, 'masks/', idx)
        #masks[idx] = mask_creation_v3(query, 'masks/', idx)
        idx += 1

    """if True:
        GT_mask = glob.glob('images/qsd1_w4/000*.png')  # Load maks from the ground truth
        GT_mask.sort()
        print(GT_mask)

        mean_precision = []
        mean_recall = []
        mean_f1score = []

        for i in range(0, len(GT_mask)):  # For each pair of masks, obtain the recall, precision and f1score metrics
            recall, precision, f1score = evaluate_mask(cv2.cvtColor(cv2.imread(GT_mask[i]), cv2.COLOR_BGR2GRAY),
                                                                    masks[i], idx)
            mean_recall.append(recall)
            mean_precision.append(precision)
            mean_f1score.append(f1score)

        print('Recall: ' + str(np.array(mean_recall).mean()))
        print('Precision: ' + str(np.array(mean_precision).mean()))
        print('F1 score: ' + str(np.array(mean_f1score).mean()))"""