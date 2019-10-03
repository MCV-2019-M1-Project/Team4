import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

######################## Function to calculate 1D grayscale histogram ##########################################333

def calculateValidationHistogram1D(image_name): 

    # read image from the input parameter
    image = os.path.join("../images/", image_name, ".jpg") # Relative path to open the images from folder utils
        
    # Compute the histogram for the query image
    im = cv2.imread(image) 
    hist, bins = np.histogram(im.ravel(),256,[0,256])

    return hist # Histogram output with shape [1, number of pixel(0-256)]


######################## Function to calculate 3 BGR histograms ##########################################333

def calculateValidationHistogramBGR(image_name): 

    # read image from the parameter
    image = os.path.join("../images/", image_name, ".jpg")
    hist_global = [] # Declaration of the return list with the following shape [3, number of pixel(0-256)]

    im = cv2.imread(image)

    # calculate and plot histogram using BGR

    color = ('b','g','r')
    for i,col in enumerate(color):
        hist = cv2.calcHist([im],[i],None,[256],[0,256])
        hist_global.append(hist)

    return hist_global


####################### Function to calculate 3 LAB histograms ##########################################333


def calculateValidationHistogramLAB(image_name):

    # read image from the parameter
    image = os.path.join("../images/", image_name, ".jpg")
    hist_global = []

    im = cv2.imread(image)

    # calculate and plot histogram using LAB color-space
    lab = ('L','b','a')
    for i,col in enumerate(lab):
        hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2LAB)],[i],None,[256],[0,256])
        hist_global.append(hist)

    return hist_global


######################## Function to calculate 3 YCrCb histograms ##########################################333


def calculateValidationHistogramYCrCb(image_name):

    # read image from the parameter
    image = os.path.join("../images/", image_name, ".jpg")
    hist_global = []

    im = cv2.imread(image)

    # calculate and plot histogram using YCrCb color-space
    lab = ('Y','Cr','b')
    for i,col in enumerate(lab):
        hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)],[i],None,[256],[0,256])
        hist_global.append(hist)
        
    return hist_global


######################## Function to calculate 3 HSV histograms ##########################################333

def calculateValidationHistogramHSV(image_name):

    # read image from the parameter
    image = os.path.join("../images/", image_name, ".jpg")
    hist_global = []

    im = cv2.imread(image)

    # calculate and plot histogram using HSV color-space
    lab = ('H','S','V')
    for i,col in enumerate(lab):
        hist = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)],[i],None,[256],[0,256])
        hist_global.append(hist)

    return hist_global



    
    