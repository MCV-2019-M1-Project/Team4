import cv2
import numpy as np
from matplotlib import pyplot as plt

image_bbdd = cv2.imread("images/qsd2_w1/00000.jpg")
image_TO_COMPARE = cv2.imread('images/bbdd/bbdd_00228.jpg')
image_TO_COMPARE = cv2.cvtColor(image_TO_COMPARE, cv2.COLOR_BGR2GRAY)
mask = cv2.imread("images/qsd2_w1/00000.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask2 = cv2.imread("masks/a00_mask.png")
mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

mask = np.uint8(mask)
mask2 = np.uint8(mask2)

hist = cv2.calcHist([image_bbdd], [0], mask, [256], [0, 256])
hist2 = cv2.calcHist([image_bbdd], [0], mask2, [256], [0, 256])
hist_to_compare = cv2.calcHist([image_bbdd], [0], None, [256], [0, 256])
masked_img = cv2.bitwise_and(image_bbdd, image_bbdd, mask = mask2)

plt.subplot(221), plt.imshow(image_bbdd, 'gray')
plt.subplot(222), plt.imshow(mask2,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_to_compare), plt.plot(hist2)
plt.xlim([0, 256])

plt.show()