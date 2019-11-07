# import nmslib as nmslib
import numpy as np
import cv2

def _filter_matches(matches, ratio=0.5):
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _compute_similarity_score(matches, matches_thresh, dist_thresh=920):
    m = len(matches)
    d = np.mean([match.distance for match in matches]) if m > 0 else np.inf
    if m < matches_thresh or d > dist_thresh:
        return 0
    else:
        return m / d


def bf_match(query_des, image_des, distance_metric, threshold):
    norm_type = {
        'l1': cv2.NORM_L1,
        'l2': cv2.NORM_L2,
        'hamming': cv2.NORM_HAMMING,
        'hamming2': cv2.NORM_HAMMING2
    }
    bf = cv2.BFMatcher(normType=norm_type[distance_metric])

    # For each image descriptor, find best k matches among query descriptors
    matches = bf.knnMatch(image_des, query_des, k=2)
    good = _filter_matches(matches)
    score = _compute_similarity_score(good, threshold)

    return score


def flann_match(query_des, image_des, distance_metric, threshold):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each image descriptor, find best k matches among query descriptors
    matches = flann.knnMatch(image_des, query_des, k=2)
    good = _filter_matches(matches)
    score = _compute_similarity_score(good, threshold)

    return score


def nmslib_match(query_des, image_des, distance_metric, threshold):
    index = nmslib.init(method='hnsw', space='l2sqr_sift', data_type=nmslib.DataType.DENSE_UINT8_VECTOR, dtype=nmslib.DistType.INT)
    index.addDataPointBatch(query_des.astype(np.uint8))
    index.createIndex({'M': 16, 'efConstruction': 100})
    index.setQueryTimeParams({'efSearch': 100})

    # For each image descriptor, find best k matches among query descriptors
    matches = index.knnQueryBatch(image_des.astype(np.uint8), k=2)
    matches = [[cv2.DMatch(query_idx, train_idx, distance) for train_idx, distance in zip(*match)] for query_idx, match in enumerate(matches)]
    good = _filter_matches(matches)
    score = _compute_similarity_score(good, threshold)

    return score


def match_descriptors(query_des, image_des, method, distance_metric, threshold):
    func = {
        'brute_force': bf_match,
        'flann': flann_match,
        'nmslib': nmslib_match
    }
    return func[method](query_des, image_des, distance_metric, threshold)