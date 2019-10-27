import os
import cv2
from text import bounding_boxes_detection
import glob

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def extract_text(cropped_image_path):
    """
    This function applies the OCR to get the text from paintings
    :param cropped_image_path:
    :return:
    """

    return pytesseract.image_to_string(Image.open(cropped_image_path))


def get_text(img_path, mask_text_path, method, idx, x_pixel_to_split, side):
    """
    This function returns the detected text after recognition
    :param img_path:
    :param mask_text_path:
    :param method:
    :param idx:
    :param x_pixel_to_split: indicates the x pixel to split the image and mask if there are more than one painting
    :param side: indicates the side to split the image and mask if there are more than one painting
    :return:
    """

    _ = bounding_boxes_detection(img_path, mask_text_path, method, True, idx)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_text_path + str(idx) + '.png')

    if x_pixel_to_split is not None:
        if side == "left":
            image = image[1:image.shape[0], 1:int(x_pixel_to_split)]
            mask = mask[1:mask.shape[0], 1:int(x_pixel_to_split)]

        elif side == "right":
            image = image[1:image.shape[0], int(x_pixel_to_split):image.shape[1]]
            mask = mask[1:mask.shape[0], int(x_pixel_to_split):mask.shape[1]]

    # cv2.imwrite(img_path.replace(".jpg", "_boxmask.png"), cv2.resize(mask_image,(1000,1000)))
    
    text_img = cv2.bitwise_and(image, mask)
    # cropped_text = image[text_boxes[idx][0][1] : text_boxes[idx][0][3], text_boxes[idx][0][0] : text_boxes[idx][0][2]]
    # text_img_path = img_path.replace(".jpg", "_text.png")
    # cv2.imwrite(img_path + str(idx) + '_text.png', text_img)

    text = extract_text(img_path + str(idx) + '_text.png')

    # text = detect_text(img_path.replace(".jpg","_denoised.png"))
    return text





