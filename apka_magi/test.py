import numpy as np
import cv2
im = cv2.imread('/home/wdenejko/PracaMagisterska-Github/apka_magi/Latin/A/58a9cea0bee57.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im3 = cv2.drawContours(im2, contours, -1, (0,255,0), 3)

cv2.imwrite('/home/wdenejko/PracaMagisterska-Github/apka_magi/test.png',im3)