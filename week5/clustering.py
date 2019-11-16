import glob
import cv2
from week5.texture_descriptors import *
from week5.histogram import *
import sys
import numpy as np
from sklearn.cluster import KMeans
from skimage import feature 
import os
import shutil
from tqdm import tqdm
import time


def lbp_texture_clustering(images):
    """

    :return:
    """

    # images = [cv2.imread(img) for img in filenames]
    # print(len(filenames))

    # image_features = {} # Store the image features in a dict
    # idx = 0
    # texture_method = 'LBP'
    # texture_descriptor_level = 3
    # settings for LBP
    radius = 3
    n_points = 8 * radius
    feat = []
    lbp_feat = []
    temp_feat = []
    fn = np.array([[]])
    for fn in tqdm(images):
        # temp_feat = get_image_texture_descriptor(fn, texture_method, texture_descriptor_level, None, None, None)
        # image_features[fn] = np.concatenate(temp_feat, axis=0)
        im = cv2.imread(fn)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        lbp_feat = feature.local_binary_pattern(im_gray, n_points, radius, method="uniform")
        # n_bins = 256
        temp_feat, _ = np.histogram(lbp_feat.ravel(), density=True, bins=n_points+2, range = (0, n_points+2))
        # temp_feat_new = np.asarray(temp_feat)
        # import pdb; pdb.set_trace()
        # print(temp_feat)
        feat.append(temp_feat)
        # sys.exit(0)
        # idx += 1
    feat_np = np.asarray(feat)
    print("performing KMeans")
    kmeans = KMeans(n_clusters=10, random_state=None).fit(feat_np)

    # print(kmeans.shape)
    # print(kmeans.labels_)

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/josep/Code/M1/week5/Clustering/LBP/'+str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)

    print(kmeans.cluster_centers_)


def hog_texture_clustering(images):

    features = []

    for image in tqdm(images):
        im = cv2.imread(image)
        hog = HOG_descriptor(im, None)
        features.append(hog)

    print("\n Performing KMeans")
    kmeans = KMeans(n_clusters=10, random_state=None).fit(np.array(features))

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/josep/Code/M1/week5/Clustering/HOG/' + str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)


def colour_clustering(images):

    features = []

    for image in tqdm(images):
        img = cv2.imread(image)
        img = cv2.resize(img, (512, 512))
        features.append(img.flatten())

    print("\n Performing KMeans")
    kmeans = KMeans(n_clusters=10, random_state=None).fit(np.array(features))

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/josep/Code/M1/week5/Clustering/Colour/'+str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)


def combined_clustering(images):

    features = []

    for image in tqdm(images):
        im = cv2.imread(image)
        im = cv2.resize(im, (256, 256))
        feature = im.flatten()
        hog = HOG_descriptor(im, None)
        feature = feature.tolist()
        feature.extend(hog)
        features.append(feature)

    print("\n Performing KMeans")
    kmeans = KMeans(n_clusters=10, random_state=None).fit(np.array(features))

    for i, k_l in enumerate(kmeans.labels_):
        dst_dir = '/home/josep/Code/M1/week5/Clustering/Combined/' + str(k_l)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(filenames[i], dst_dir)


if __name__ == "__main__":
    filenames = glob.glob("images/bbdd/*.jpg")
    filenames.sort()

    print("Starting LBP clustering")
    time_start = time.time()
    lbp_texture_clustering(filenames)
    time_end = time.time()
    print("Elapsed time: ", time_end - time_start)

    print("Starting HOG clustering")
    time_start = time.time()
    hog_texture_clustering(filenames)
    time_end = time.time()
    print("Elapsed time: ", time_end - time_start)

    print("Starting Colour clustering")
    time_start = time.time()
    colour_clustering(filenames)
    time_end = time.time()
    print("Elapsed time: ", time_end - time_start)

    print("Starting Combined clustering")
    time_start = time.time()
    combined_clustering(filenames)
    time_end = time.time()
    print("Elapsed time: ", time_end - time_start)


