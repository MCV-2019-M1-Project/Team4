import imutils
import numpy as np
import cv2
import glob
from week5.mask import *


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


def find_paintings(image_path, masks_path, idx):
    """

    :param image_path:
    :param masks_path:
    :param idx:
    :return:
    """

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = mask_creation_v2(image_path, masks_path, idx)


    # find contours in the thresholded image and initialize the
    # shape detector
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    sd = ShapeDetector()

    bbox = []
    # loop over the contours
    for c in contours:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * 1)
        cY = int((M["m01"] / M["m00"]) * 1)
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= 1
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    cv2.imshow("result", cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cropped = []
    return mask, cropped


if __name__ == '__main__':
    query_filenames = glob.glob("images/qsd1_w5_denoised/*.jpg")
    query_filenames.sort()
    cropped = {}
    masks = {}
    print(query_filenames)
    idx = 0

    for query in query_filenames:
        if idx == 0 or idx == 3 or idx == 7:
            # pass
            masks[idx], cropped[idx] = find_paintings(query, 'masks/', idx)
        # masks[idx] = mask_creation_v3(query, 'masks/', idx)
        idx += 1
