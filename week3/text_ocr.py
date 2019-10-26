import os
from glob import glob
import cv2

from text import bounding_boxes_detection



import pytesseract


 

def extract_text(text_img_path):
   
    '''This function applies the OCR to get the text from paintings'''

    return pytesseract.image_to_string(Image.open(text_img_path))


def get_text(img,img_path, method):
   
    ''' This function returns the detected text after recognition '''
  
    print("Detecting textbox of images",img_path)
    text_mask, text_box = bounding_boxes_detection(img, method)
    bbox = [text_box[0][1], text_box[0][0], text_box[1][1], text_box[1][0]]
    cv2.imwrite(img_path.replace(".jpg", "_boxmask.png"), cv2.resize(text_mask,(1000,1000)))
    
    text_img = cv2.bitwise_and(img, img, mask = text_mask)
    cropped_text = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    text_img_path = img_path.replace(".jpg", "_text.png")
    cv2.imwrite(text_img_path, cropped_text)
    text = extract_text(text_img_path)

    # text = detect_text(img_path.replace(".jpg","_denoised.png"))
    return text





