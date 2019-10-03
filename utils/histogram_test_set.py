import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

######################## Function to calculate 1D grayscale histogram ##########################################333

def calculateTestHistogram1D(): 

    # read images in dataset
    filenames = glob.glob("../images/bbdd/*.jpg") # Relative path to open the images from folder utils
    hist_1D_global = [] # Declaration of the return list with the following shape [number of images, number of pixel(0-256)]

    for image in filenames: # Loop for each image

        # Compute the histogram for every image
        im = cv2.imread(image)
        hist, bins = np.histogram(im.ravel(),256,[0,256])

        # Append all histograms together to the list
        hist_1D_global.append(hist)

    return hist_1D_global


######################## Function to calculate 3 BGR histograms ##########################################333

def calculateTestHistogramBGR(): 

    # read images in dataset
    filenames = glob.glob("../images/bbdd/*.jpg")
    hist_global = [] # Declaration of the return list with the following shape [3 * number of images, number of pixel(0-256)]

    for image in filenames:

        im = cv2.imread(image)

        color = ('b','g','r')
                
        # Compute the histogram for every image and color (3 for every image in total)

        for i,col in enumerate(color):
            hist = cv2.calcHist([im],[i],None,[256],[0,256])
            hist_global.append(hist)

    return hist_global


####################### Function to calculate 3 LAB histograms ##########################################333


def calculateTestHistogramLAB():

    # read images in dataset
    filenames = glob.glob("../images/bbdd/*.jpg")
    hist_global = []

    for image in filenames:

        im = cv2.imread(image)

    # calculate and plot histogram using LAB color-space
    lab = ('L','b','a')
    for i,col in enumerate(lab):
        hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2LAB)],[i],None,[256],[0,256])
        hist_global.append(hist)

    return hist_global


######################## Function to calculate 3 YCrCb histograms ##########################################333


def calculateTestHistogramYCrCb():

    # read images in dataset
    filenames = glob.glob("../images/bbdd/*.jpg")
    hist_global = []

    for image in filenames:

        im = cv2.imread(image)

    # calculate and plot histogram using YCrCb color-space
    lab = ('Y','Cr','b')
    for i,col in enumerate(lab):
        hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)],[i],None,[256],[0,256])
        hist_global.append(hist)
        
    return hist_global


######################## Function to calculate 3 HSV histograms ##########################################333

def calculateTestHistogramHSV():

    # read images in dataset
    filenames = glob.glob("../images/bbdd/*.jpg")
    hist_global = []

    for image in filenames:

        im = cv2.imread(image)

    # calculate and plot histogram using HSV color-space
        lab = ('H','S','V')
        for i,col in enumerate(lab):
            hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)],[i],None,[256],[0,256])
            hist_global.append(hist)

    return hist_global