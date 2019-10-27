import cv2
from text import bounding_boxes_detection

# import pytesseract
from PIL import Image


def extract_text(cropped_image_path, idx, to_save):
    """
    This function applies the OCR to get the text from paintings
    :param cropped_image_path:
    :param idx:
    :param to_save:
    :return:
    """

    text = pytesseract.image_to_string(cropped_image_path)

    if to_save:
        with open('results/qst1/method1/text_' + str(idx) + '.txt', 'w+') as file:
            file.writelines(text)
            file.close()

    return text


def get_text(img_path, mask_text_path, method, idx, x_pixel_to_split, side, save_text):
    """
    This function returns the detected text after recognition
    :param img_path:
    :param mask_text_path:
    :param method:
    :param idx:
    :param x_pixel_to_split: indicates the x pixel to split the image and mask if there are more than one painting
    :param side: indicates the side to split the image and mask if there are more than one painting
    :param save_text:
    :return:
    """

    text_boxes = bounding_boxes_detection(img_path, mask_text_path, method, True, idx)

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
    #cropped_text = image[text_boxes[idx][0][1]:text_boxes[idx][0][3], text_boxes[idx][0][0]:text_boxes[idx][0][2]]
    # text_img_path = img_path.replace(".jpg", "_text.png")
    cv2.imwrite('text/text_masks/' + str(idx) + '_text.png', text_img)

    text = extract_text(text_img, idx, save_text)

    # text = detect_text(img_path.replace(".jpg","_denoised.png"))
    return text





