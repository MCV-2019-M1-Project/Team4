import numpy as np
import cv2
import glob
from week5.mask import *


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
    edges = cv2.Canny(gray, 30, 100)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, np.array([]), 50, 5)

    # iterate over the output lines and draw them
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 3)

    #cv2.imshow("result", cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
    #cv2.waitKey(0)



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
