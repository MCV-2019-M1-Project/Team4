import glob
import cv2
from texture_descriptors import *
from evaluation import *
import sys
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil


filenames = glob.glob("images/bbdd/*.jpg")
filenames.sort()

museum_histograms = []
museum_textures = []

color_base = "LAB"
dimension = '2D'
metric = "bhattacharya_distance"
level = 3

texture_descriptor_level = 3
texture_method = "DCT"

idx = 0

for museum_image in filenames:
    print("Computing image", idx)

    museum_histograms.append(calculate_image_histogram(museum_image, None, color_base, dimension, level, None,
                                                        None))
    museum_textures.append(get_image_texture_descriptor(museum_image, texture_method, texture_descriptor_level,
                                                                None, None, None))
    idx += 1

feat_np = np.asarray(museum_textures, dtype=np.float64)
kmeans = KMeans(n_clusters=10, random_state=10).fit(feat_np)

for i, k_l in enumerate(kmeans.labels_):
    dst_dir = '../clustering/'+str(k_l)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(filenames[i], dst_dir)

    
print(kmeans.cluster_centers_)


