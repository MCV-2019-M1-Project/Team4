import pickle

import cv2   
import numpy as np   
import glob
import os


def bounding_boxes_detection(path):
    """
    This function detects the bounding boxes of the text in all the images of a specific folder

    :param path: path of the images
    :return: list of bounding boxes from first image to last image. Each image contains a maximum of 2 bounding boxes.

        [[[first_bounding_box_of_first_image],[second_bounding_box_of_second_image]], [[first_bounding_box_of_second_image]], ...]

    Each bounding box has the following int values:

        [lowest_pixel_x, lowest_pixel_y, highest_pixel_x, highest_pixel_y] 
    
    """

    # Open folder that contains the images
    query_filenames = glob.glob(path + '00' + '*.jpg')
    query_filenames.sort()

    # Create the empty list to store the bounding boxes coordinates
    boxes = []
    idx = 0

    # Read every image
    for query_image in query_filenames:
        
        print("Getting Text for Query Image " + str(idx))
        image = cv2.imread(query_image)

        # Color image segmentation to create binary image (255 white: high possibility of text; 0 black: no text)
        im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(im_hsv)

        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_grey[s < 3] = 255
        image_grey[image_grey != 255] = 0

        # Cleaning image using morphological opening filter
        opening_kernel = np.ones((15,10),np.uint8)
        image_grey = cv2.morphologyEx(image_grey, cv2.MORPH_OPEN, opening_kernel, iterations=1)

        # Finding contours of the white areas of the images (high possibility of text)
        contours, _ = cv2.findContours(image_grey, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # Initialize parameters
        image_width = image_grey.shape[0]
        largest_area, second_largest_area, x_box_1, y_box_1, w_box_1, h_box_1 = 0, 0, 0, 0, 0, 0

        # From all the contours found, pick only the ones with rectangular shape and large area
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if ((w/h > 2) & (w/h < 12) & (w > (0.1 * image_width)) & (area > second_largest_area)):

                if area > largest_area:
                    x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1
                    x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                    second_largest_area = largest_area
                    largest_area = area
                else:
                    x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                    second_largest_area = area
        
        # Append the corners of the bounding boxes to the boxes list
        if ((x_box_2 == y_box_2 == 0) | (path == 'images/qsd1_w2/')):
            box = [[x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]]
            boxes.append(box)
        elif x_box_1 < x_box_2:
            box = [[x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1], [x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2]]
            boxes.append(box)
        else:
            box = [[x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2], [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]]
            boxes.append(box)
        
        idx += 1

    return boxes




def bounding_boxes_evaluation(boxA, boxB):
    """
    This function evaluates the accuracy of the result bounding boxes by calculating the parameter intersection over Union (IoU)
    
    :param boxA: Ground Truth bounding boxes
    :param boxB: bounding boxes detected in the images

    :return: float with IoU parameter

    """

    iou_total = []

    for idx in range(len(boxA)):
        for subidx in range(len(boxA[idx])):
            if len(boxB[idx]) > subidx:
                
                # determine the (x, y)-coordinates of the intersection rectangle
                xA = max(boxA[idx][subidx][0], boxB[idx][subidx][0])
                yA = max(boxA[idx][subidx][1], boxB[idx][subidx][1])
                xB = min(boxA[idx][subidx][2], boxB[idx][subidx][2])
                yB = min(boxA[idx][subidx][3], boxB[idx][subidx][3])

                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (boxA[idx][subidx][2] - boxA[idx][subidx][0] + 1) * (boxA[idx][subidx][3] - boxA[idx][subidx][1] + 1)
                boxBArea = (boxB[idx][subidx][2] - boxB[idx][subidx][0] + 1) * (boxB[idx][subidx][3] - boxB[idx][subidx][1] + 1)

                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                iou_total.append(iou)

    iou_mean = sum(iou_total) / len(iou_total)

    return iou_mean