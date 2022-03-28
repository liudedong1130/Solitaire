import cv2
from matplotlib import pyplot as plt
import numpy as np



img = cv2.imread('11.PNG')
ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite('11.PNG',img)