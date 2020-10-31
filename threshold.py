import cv2
import numpy as np
from matplotlib import pyplot as plt

INPUT_FILE  = "./dataset/png/TS01_07_output.png"
OUTPUT_FILE = "./dataset/png/TS01_07_thresh.png"
NEW_SIZE = 1000

img = cv2.imread(INPUT_FILE)

# ------ resize -------
print('Resizing image...')
(height, width) = img.shape[:2]

longest_side = width
if (height > width): longest_side = height

ratio = NEW_SIZE / longest_side
img = cv2.resize(img, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_CUBIC)


# ----- create threshold -----
print('Thresholding...')
thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#_, thresh = cv2.threshold(thresh, 170, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 5)


# ----- morphs -----
print('Morphing...')
closing_kernel = np.ones((4,4),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, closing_kernel)

cv2.imshow("swaggy", opening)
cv2.waitKey(0)

dilate_kernel = np.ones((5, 8), np.uint8)
dilation = cv2.dilate(opening, dilate_kernel, iterations = 1)

cv2.imshow("swaggy", dilation)
cv2.waitKey(0)


# ----- find contours -----
contours, hierarchy = cv2.findContours(dilation,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

for c in contours:
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("swaggy", img)
cv2.waitKey(0)


# ----- save -----
#print('Saving output files...', OUTPUT_FILE)
#cv2.imshow("swaggy", thresh)
#cv2.waitKey(0)

print('Done.')