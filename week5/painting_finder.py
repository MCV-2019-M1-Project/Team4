import imutils
import numpy as np
import cv2
import glob
from scipy import ndimage

# VSCode imports
from mask import *

# PyCharm Imports
# from week5.mask import *


def find_paintings(image_path, masks_path, image_idx, query_set_path):
    """
    This function gets the bounding boxes of the paintings in an image, as well as their angle of inclination.

    :param image_path: path where the image is stored
    :param masks_path: path where the masks are stored
    :param image_idx: Index of the image
    :param query_set_path path of the query set images
    :return: quadruple containing: binary mask, list with cropped paintings, list with angle of inclination and
    bounding boxes, number of subpaintings detected
    """

    image = cv2.imread(image_path)
    mask = mask_creation_v2(image_path, masks_path, image_idx)

    # find contours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    painting_data = []
    cropped_paintings = []

    # loop over the contours
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        theta = rect[2]

        if 0.0 >= theta > -45.0:
            theta_return = theta*-1.0
        elif theta < -45.0:
            theta_return = theta*-1.0 + 90.0

        # the order of the box points: bottom left, top left, top right, bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        #cv2.imshow("result", image)
        #cv2.waitKey(0)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened

        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, transformation_matrix.astype('float32'), (width, height))

        if theta_return > 90:
            # rotation angle in degree
            warped = ndimage.rotate(warped, 90)

        #cv2.imshow("crop_img.jpg", cv2.resize(warped, (0, 0), fx=0.5, fy=0.5))
        #cv2.waitKey(0)
        box1 = box[0][0], box[0][1]
        box2 = box[1][0], box[1][1]
        box3 = box[2][0], box[2][1]
        box4 = box[3][0], box[3][1]

        aux_bbox = [box1, box2, box3, box4]
        painting_data.append([theta_return, aux_bbox])
        cropped_paintings.append([box2, warped])  # We append box2 so we can then sort the cropped paintings

    # Sort images left to right, top to bottom
    painting_data.sort(key=lambda x: x[1][1][0] + x[1][1][1])
    cropped_paintings.sort(key=lambda x: x[0][0] + x[0][1])
    cropped_paintings_return = []

    sub_image_idx = 0
    for painting in cropped_paintings:
        cropped_paintings_return.append(painting[1])
        cv2.imwrite(query_set_path + '_cropped_images/' + str(image_idx).zfill(2) + '_' + str(sub_image_idx) + ".jpg",
                    painting[1])
        sub_image_idx += 1
    del cropped_paintings

    if image_idx == 13:
        sub_image_idx = 1

    return mask, cropped_paintings_return, painting_data, sub_image_idx


if __name__ == '__main__':
    query_filenames = glob.glob("images/qsd1_w5_denoised/*.jpg")
    query_filenames.sort()
    cropped = {}
    masks = {}
    print(query_filenames)
    idx = 0

    for query in query_filenames:
        if idx == 0 or idx == 7 or idx == 16:
            # pass
            masks[idx], cropped[idx], _, _ = find_paintings(query, 'masks/', idx, 'qsd1_w5')
        # masks[idx] = mask_creation_v3(query, 'masks/', idx)
        idx += 1
