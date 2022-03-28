import cv2
from matplotlib import pyplot as plt
import numpy as np


def number_con(img):
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img", img_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret, img_3 = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("img", img_3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((3, 3), np.uint8)
    img_3 = cv2.dilate(img_3, kernel, iterations=2)
    img_3 = cv2.erode(img_3, kernel, iterations=1)
    img_3 = cv2.dilate(img_3, kernel, iterations=6)
    # cv2.imshow("img", img_3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(img_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # draw_img = img.copy()
    # res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
    # cv2.imshow('res',res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    this_number = 0
    if hierarchy is None:
        return np.array([-1]), np.array([[0, 0]])
    for i in range(hierarchy.shape[1]):
        cnt = contours[i]
        if cv2.contourArea(cnt) > 500:
            this_number = this_number + 1
    number = np.zeros(this_number)
    position = np.zeros(2)
    list_position = np.tile(position, (this_number, 1))
    r = 0
    for i in range(hierarchy.shape[1]):
        cnt = contours[i]
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # img_2 = cv2.rectangle(img_2,(x,y),(x+w,y+h),(0,255,0),2)
        number_1 = img[y:(y + 36), 0:37]
        # cv2.imshow("img", number_1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        similarity = np.zeros(13)
        for m in range(13):

            # img1 = cv2.imread('6.PNG')
            img2 = cv2.imread(str(m + 1) + '.PNG')
            ret, number_1 = cv2.threshold(number_1, 50, 255, cv2.THRESH_BINARY)

            # ret, img2 = cv2.threshold(img2, 254, 255, cv2.THRESH_TRUNC)
            # cv2.imshow("img", number_1)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(img1.shape)
            distance = np.zeros((3, 5))
            for h in range(-1, 2):
                for g in range(-2, 3):
                    for p in range(2, 22):
                        for j in range(8, 22):
                            distance[h + 1, g + 2] = distance[h + 1, g + 2] + abs(
                                int(number_1[(p - h), (j - g), 0]) - int(img2[p, j, 0]))
            similarity[m] = distance.min()
        number[r] = np.argmin(similarity) + 1
        list_position[r] = [x, y]
        r = r + 1
    return number, list_position


