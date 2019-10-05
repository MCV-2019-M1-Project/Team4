import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

######################## Function to calculate 1D grayscale histogram ##########################################333


def calculateHistogram1D(filenames, path):

    hist_1D_global = {} # Declaration of the return list with the following shape [number of images, number of pixel(0-256)]

    idx = 0
    for image in filenames: # Loop for each image

        # Compute the histogram for every image
        im = cv2.imread(image)
        hist, bins = np.histogram(im.ravel(),256,[0,256])

        # Append all histograms together to the list
        hist_1D_global[idx] = hist
        idx += 1

    return hist_1D_global


######################## Function to calculate 3 BGR histograms ##########################################333


def calculateHistogramBGR(filenames, path):


    hist_image_set = {} # Declaration of the return list with the following shape [3 * number of images, number of pixel(0-256)]

    idx = 0
    for image in filenames:

        im = cv2.imread(image)

        color = ('b','g','r')
                
        # Compute the histogram for every image and color (3 for every image in total)
        hist_image = []
        for i,col in enumerate(color):
            hist = cv2.calcHist([im],[i],None,[256],[0,256])
            hist_image.append(hist)

        hist_image_set[idx] = hist_image
        idx += 1

    return hist_image_set


####################### Function to calculate 3 LAB histograms ##########################################333


def calculateHistogramLAB(filenames, path):

    hist_image_set = {}

    idx = 0
    for image in filenames:

        im = cv2.imread(image)

        # calculate and plot histogram using LAB color-space
        lab = ('L','b','a')
        hist_image = []
        for i, col in enumerate(lab):
            hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2LAB)],[i], None,[256],[0, 256])
            hist_image.append(hist)

        hist_image_set[idx] = hist_image
        idx += 1

    return hist_image_set


######################## Function to calculate 3 YCrCb histograms ##########################################333


def calculateHistogramYCrCb(filenames, path):

    hist_image_set = []

    idx = 0
    for image in filenames:

        im = cv2.imread(image)

        # calculate and plot histogram using YCrCb color-space
        lab = ('Y','Cr','b')
        hist_image = []

        for i,col in enumerate(lab):
            hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)],[i],None,[256],[0,256])
            hist_image.append(hist)

        hist_image_set[idx] = hist_image
        idx += 1


    return hist_image_set


######################## Function to calculate 3 HSV histograms ##########################################333


def calculateHistogramHSV(filenames, path):

    hist_image_set = []

    idx = 0
    for image in filenames:

        im = cv2.imread(image)

        # calculate and plot histogram using HSV color-space
        lab = ('H','S','V')
        hist_image = []

        for i,col in enumerate(lab):
            hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)],[i],None,[256],[0,256])
            hist_image.append(hist)

        hist_image_set[str(idx)] = hist_image
        idx += 1

    return hist_image_set


def calculateHistogram(path, colorBase):

    # read images in dataset
    filenames = glob.glob(path + "*.jpg")
    filenames.sort()

    func = {
        'HSV': calculateHistogramHSV,
        'YCrCb': calculateHistogramYCrCb,
        'LAB': calculateHistogramLAB,
        'BGR': calculateHistogramBGR,
        '1D': calculateHistogram1D,
    }
    return func[colorBase](filenames, path)
