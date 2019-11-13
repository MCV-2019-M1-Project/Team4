import pickle

import cv2   
import numpy as np   
import glob
import os
import argparse
import time
from imutils.object_detection import non_max_suppression


def bounding_boxes_detection(image_path, mask_set_path, method, save_masks, subpaintings, idx):
    """
    This function detects the bounding boxes of the text in all the images of a specific folder

    :param image_path: path of the images
    :param mask_set_path: path where the masks will be saved
    :param method: 1 for color segmentation, 2 for morphology operations, 3 for neural network
    :param save_masks: bool indicating if the masks need to be saved
    :param subpaintings: compute if there are more than one subpaintings in the image: 1, 2 or 3
    :param idx: int containing the index of the image
    :return: list of bounding boxes from first image to last image. Each image contains a maximum of 2 bounding boxes.

        [[[first_bounding_box_of_first_image],[second_bounding_box_of_second_image]], [[first_bounding_box_of_second_image]], ...]

    Each bounding box has the following int values:

        [lowest_pixel_x, lowest_pixel_y, highest_pixel_x, highest_pixel_y] 
    
    """

    # Create the empty list to store the bounding boxes coordinates
    boxes = []
    # Read every image
    image = cv2.imread(image_path)

    #----------------------------------   METHOD 1   ----------------------------------------------------------
    """
    Method 1: text detection based on color segmentation using saturation
    """
    if method == 1:

        saturation_threshold = 20

        # Color image segmentation to create binary image (255 white: high possibility of text; 0 black: no text)
        im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(im_hsv)

        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_grey[s < saturation_threshold] = 255
        image_grey[image_grey != 255] = 0

        # Cleaning image using morphological opening filter
        opening_kernel = np.ones((5, 5), np.uint8)/9
        text_mask = cv2.morphologyEx(image_grey, cv2.MORPH_OPEN, opening_kernel, iterations=1)


    #----------------------------------   METHOD 2   ----------------------------------------------------------
    """
    Method 2: text detection based on morphology operations
    """

    if method == 2:

        # Define grayscale image
        im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        im_y, _, _ = cv2.split(im_yuv)

        # Define kernel sizes
        kernel = np.ones((5, 10), np.float32)/9

        # Difference between erosion and dilation images
        y_dilation = cv2.morphologyEx(im_y, cv2.MORPH_DILATE, kernel, iterations=1)
        y_erosion = cv2.morphologyEx(im_y, cv2.MORPH_ERODE, kernel, iterations=1)

        difference_image = y_erosion - y_dilation

        # Grow contrast areas found
        growing_image = cv2.morphologyEx(difference_image, cv2.MORPH_ERODE, kernel, iterations=1)

        # Low pass filter to smooth out the result
        blurry_image = cv2.filter2D(growing_image, -1, kernel)

        # Thresholding the image to make a binary image
        ret, binary_image = cv2.threshold(blurry_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted_binary_image = cv2.bitwise_not(binary_image)

        # Clean small white pixels areas outside text using closing filter
        #text_mask = cv2.morphologyEx(inverted_binary_image, cv2.MORPH_OPEN, kernel, iterations = 1)

        text_mask = inverted_binary_image

    #----------------------------------   METHOD 3   ----------------------------------------------------------
    """
    Method 3: EAST opencv text detection based on Neural Network
    """
    if method == 3:

        min_confidence = 0.8

        # load the input image and grab the image dimensions
        image = cv2.imread(image_path)
        orig = image.copy()
        (H, W) = image.shape[:2]

        # create an empty text mask
        text_mask = np.zeros((H,W,1), np.uint8)

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        newW, newH = 320, 320
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        # print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet("text/frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction
        # print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            bounding_boxes = non_max_suppression(np.array(rects), probs=confidences)

        # merge closed bounding boxes
        for idx1, box1 in enumerate(bounding_boxes):
            for idx2, box2 in enumerate(bounding_boxes):
                if (abs(box1[3] - box2[3]) < 10) and (abs(box1[0] - box2[2]) < 30 or abs(box1[2] - box2[0]) < 30):
                    bounding_boxes[idx1] = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
                    bounding_boxes[idx2] = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in bounding_boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # if ((endX - startX) / (endY - startY) > 2) & ((endX - startX) / (endY - startY)  < 12) & ((endX - startX) > (0.1 * W)):
        
            # draw the bounding box on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # box = [startX, startY, endX, endY]
            # bounding_boxes.append(box)
            text_mask[startY - 5 : endY + 5, startX - 5  : endX + 5] = 255


    #------------------------------   FINDING AND CHOOSING CONTOURS OF THE BINARY MASK   ---------------------------------------

    # Finding contours of the white areas of the images (high possibility of text)
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Initialize parameters
    largest_area, second_largest_area, third_largest_area, x_box_1, y_box_1, w_box_1, h_box_1, x_box_2, y_box_2, w_box_2, h_box_2, x_box_3, y_box_3, w_box_3, h_box_3 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    image_width = text_mask.shape[0]
    image_area = text_mask.shape[0] * text_mask.shape[1]

    # From all the contours found, pick only the ones with rectangular shape and large area
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if (w / h > 2.5) & (w / h < 15) & (w > (0.1 * image_width)) & (area > second_largest_area) & (area < 0.1 * image_area):

            if area > third_largest_area:
                if area > second_largest_area:
                    if area > largest_area:
                        x_box_3, y_box_3, w_box_3, h_box_3 = x_box_2, y_box_2, w_box_2, h_box_2
                        x_box_2, y_box_2, w_box_2, h_box_2 = x_box_1, y_box_1, w_box_1, h_box_1   
                        x_box_1, y_box_1, w_box_1, h_box_1 = x, y, w, h
                        third_largest_area = second_largest_area
                        second_largest_area = largest_area
                        largest_area = area
                    else:
                        x_box_3, y_box_3, w_box_3, h_box_3 = x_box_2, y_box_2, w_box_2, h_box_2
                        x_box_2, y_box_2, w_box_2, h_box_2 = x, y, w, h
                        third_largest_area = second_largest_area
                        second_largest_area = area
                else:
                    x_box_3, y_box_3, w_box_3, h_box_3 = x, y, w, h
                    third_largest_area = area

    # cv2.rectangle(image, (x_box_1, y_box_1), (x_box_1 + w_box_1 - 1, y_box_1 + h_box_1 - 1), 255, 2)
    # cv2.rectangle(image, (x_box_2, y_box_2), (x_box_2 + w_box_2 - 1, y_box_2 + h_box_2 - 1), 255, 2)

    # Append the corners of the bounding boxes to the boxes list
    text_mask[:,:] = 0

    if subpaintings == 1:
        box = [[x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]]
        boxes.append(box)
        text_mask[y_box_1 : (y_box_1 + h_box_1), x_box_1 : (x_box_1 + w_box_1)] = 255

    if subpaintings == 2:
        if x_box_1 < x_box_2:
            box = [[x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1], [x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2]]
            boxes.append(box)
        else:
            box = [[x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2], [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]]
            boxes.append(box)
        text_mask[y_box_1 : (y_box_1 + h_box_1), x_box_1 : (x_box_1 + w_box_1)] = 255
        text_mask[y_box_2 : (y_box_2 + h_box_2), x_box_2 : (x_box_2 + w_box_2)] = 255

    if subpaintings == 3:
        if x_box_1 < x_box_2 < x_box_3:
            box = [[x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1], [x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2], [x_box_3, y_box_3, x_box_3 + w_box_3, y_box_3 + h_box_3] ]
            boxes.append(box)
        elif x_box_1 < x_box_3 < x_box_2:
            box = [[x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1], [x_box_3, y_box_3, x_box_3 + w_box_3, y_box_3 + h_box_3], [x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2]]
            boxes.append(box)
        elif x_box_2 < x_box_1 < x_box_3:
            box = [[x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2], [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1], [x_box_3, y_box_3, x_box_3 + w_box_3, y_box_3 + h_box_3]]
            boxes.append(box)
        elif x_box_2 < x_box_3 < x_box_1:
            box = [[x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2], [x_box_3, y_box_3, x_box_3 + w_box_3, y_box_3 + h_box_3], [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]]
            boxes.append(box)
        elif x_box_3 < x_box_1 < x_box_2:
            box = [[x_box_3, y_box_3, x_box_3 + w_box_3, y_box_3 + h_box_3], [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1], [x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2]]
            boxes.append(box)            
        elif x_box_3 < x_box_2 < x_box_1:
            box = [[x_box_3, y_box_3, x_box_3 + w_box_3, y_box_3 + h_box_3], [x_box_2, y_box_2, x_box_2 + w_box_2, y_box_2 + h_box_2], [x_box_1, y_box_1, x_box_1 + w_box_1, y_box_1 + h_box_1]]
            boxes.append(box)

        text_mask[y_box_1 : (y_box_1 + h_box_1), x_box_1 : (x_box_1 + w_box_1)] = 255
        text_mask[y_box_2 : (y_box_2 + h_box_2), x_box_2 : (x_box_2 + w_box_2)] = 255
        text_mask[y_box_3 : (y_box_3 + h_box_2), x_box_3 : (x_box_3 + w_box_3)] = 255

    if save_masks:
        cv2.imwrite(mask_set_path + str(idx) + '.png', text_mask)

    return boxes


def bounding_boxes_evaluation(boxA, boxB):
    """
    This function evaluates the accuracy of the result bounding boxes by calculating the parameter intersection over
    Union (IoU)

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

            else: 
                iou_total.append(0)

    iou_mean = sum(iou_total) / len(iou_total)

    return iou_mean
