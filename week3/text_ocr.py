import os
import cv2
from text import bounding_boxes_detection
import glob

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


def extract_text(cropped_image_path):
   
    '''This function applies the OCR to get the text from paintings'''

    return pytesseract.image_to_string(Image.open(cropped_image_path))


def get_text(img_path, mask_text_path, method):
   
    ''' This function returns the detected text after recognition '''
  
    print("Detecting textbox of images")
    
    text_boxes = bounding_boxes_detection(img_path, mask_text_path, method)

    museum_filenames = glob.glob(img_path + '*.jpg')
    museum_filenames.sort()
    text_masks_filenames = glob.glob(mask_text_path + '*.png')
    text_masks_filenames.sort()

    idx = 0

    text_list = []

    for museum_image in museum_filenames:
        image = cv2.imread(museum_image)
        mask_image = cv2.imread(text_masks_filenames[idx])
        # cv2.imwrite(img_path.replace(".jpg", "_boxmask.png"), cv2.resize(mask_image,(1000,1000)))
    
        text_img = cv2.bitwise_and(image, mask_image)
        # cropped_text = image[text_boxes[idx][0][1] : text_boxes[idx][0][3], text_boxes[idx][0][0] : text_boxes[idx][0][2]]
        # text_img_path = img_path.replace(".jpg", "_text.png")
        cv2.imwrite(img_path + str(idx) + '_text.png', text_img)
        
        text = extract_text(img_path + str(idx) + '_text.png')

        text_list.append(text)

        idx += 1
    
        # text = detect_text(img_path.replace(".jpg","_denoised.png"))
    return text_list





