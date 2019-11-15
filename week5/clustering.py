import glob
import cv2
from texture_descriptors import *
import sys
import numpy as np
from sklearn.cluster import KMeans
from skimage import feature 
import os
import shutil
from tqdm import tqdm


filenames = glob.glob("images/bbdd/*.jpg")
filenames.sort()
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
fn =  np.array([[]])
for fn in tqdm(filenames):
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
kmeans = KMeans(n_clusters=10, random_state=None).fit(feat_np)

# print(kmeans.shape)
# print(kmeans.labels_)
for i, k_l in enumerate(kmeans.labels_):
    dst_dir = '/home/sanket/Team4/week5/'+str(k_l)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(filenames[i], dst_dir)

    
print(kmeans.cluster_centers_)


