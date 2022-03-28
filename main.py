import pyautogui as pg
import cv2
import numpy as np
from time import *
import sys


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


def gettb(inf):
    t_b_number = np.zeros((10, 2))
    t_b_position = np.zeros((10, 2, 2))
    for i in range(10):
        t_b_number[i][0] = inf[i][0][0]
        t_b_position[i][0] = inf[i][1][0]
        t_b_number[i][1] = t_b_number[i][0]
        t_b_position[i][1] = t_b_position[i][0]

        for j in range(inf[i][0].shape[0] - 1):
            if inf[i][0][j + 1] == (inf[i][0][j] + 1):
                t_b_number[i][1] = inf[i][0][j + 1]
                t_b_position[i][1] = inf[i][1][j + 1]
            else:
                break
    return t_b_number, t_b_position


def fapai():
    pg.click(x=2003, y=1126)
    sleep(1)


def judgefinish(img):
    imgtest = img[1029:1062, 314:760]
    img_gray = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
    ret, img_3 = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    img_3 = cv2.dilate(img_3, kernel, iterations=2)
    img_3 = cv2.erode(img_3, kernel, iterations=1)
    img_3 = cv2.dilate(img_3, kernel, iterations=6)
    contours, hierarchy = cv2.findContours(img_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    listpos_hang = [[145, 1324], [145, 1324], [145, 1324], [145, 1324], [145, 1324], [145, 1324], [145, 1324],
                    [145, 1000],
                    [145, 1000], [145, 1000]]
    if hierarchy is None:
        return listpos_hang
    if hierarchy.shape[1] == 8:
        listpos_hang = [[145, 1006], [145, 1006], [145, 1006], [145, 1006], [145, 1324], [145, 1324], [145, 1324],
                        [145, 1000],
                        [145, 1000], [145, 1000]]
    else:
        listpos_hang = [[145, 1006], [145, 1006], [145, 1006], [145, 1324], [145, 1324], [145, 1324], [145, 1324],
                        [145, 1000],
                        [145, 1000], [145, 1000]]
    return listpos_hang


begin_time = time()
sleep(2)
listpos_lie = [[143, 180], [374, 411], [607, 644], [838, 875], [1071, 1106], [1302, 1339], [1535, 1572], [1767, 1804],
               [1999, 2036], [2231, 2268]]
photo_x = [143, 374, 607, 838, 1071, 1302, 1535, 1767, 1999, 2231]
img = pg.screenshot(region=[0, 0, 2559, 1439])  # x,y,w,h
img.save('example.PNG')
img = cv2.imread('example.PNG')
listpos_hang = judgefinish(img)
img_1 = img[145:1006, 143:180]
sb = number_con(img_1)
sb3 = [[0, 0], [[0, 0], [0, 0]]]
inf = np.array([sb, sb3, sb3, sb, sb, sb, sb, sb, sb, sb], dtype=object)
for i in range(10):
    inf[i] = number_con(img[listpos_hang[i][0]:listpos_hang[i][1], listpos_lie[i][0]:listpos_lie[i][1]])
    inf[i][1] = inf[i][1] + [photo_x[i], 145]
t_b = gettb(inf)
fapaitime = 0
exitnum = 1
while (1):
    judgemove = 0
    for i in range(10):
        for j in range(10):
            if j == i:
                continue
            if (t_b[0][i][0] < t_b[0][j][0]) and (t_b[0][i][1] < t_b[0][j][1]) and (t_b[0][i][1] >= t_b[0][j][0] - 1):
                pg.click(x=inf[i][1][int(t_b[0][j][0] - t_b[0][i][0] - 1)][0],
                         y=inf[i][1][int(t_b[0][j][0] - t_b[0][i][0] - 1)][1])
                sleep(0.1)
                pg.click(x=t_b[1][j][0][0], y=t_b[1][j][0][1])
                if t_b[0][j][1] == 13:
                    sleep(1)
                else:
                    sleep(0.5)
                pg.moveTo(100, 100)
                img = pg.screenshot(region=[0, 0, 2559, 1439])  # x,y,w,h
                img.save('example.PNG')
                img = cv2.imread('example.PNG')
                listpos_hang = judgefinish(img)
                inf[i] = number_con(img[listpos_hang[i][0]:listpos_hang[i][1], listpos_lie[i][0]:listpos_lie[i][1]])
                inf[i][1] = inf[i][1] + [photo_x[i], 145]
                inf[j] = number_con(img[listpos_hang[j][0]:listpos_hang[j][1], listpos_lie[j][0]:listpos_lie[j][1]])
                inf[j][1] = inf[j][1] + [photo_x[j], 145]
                t_b = gettb(inf)
                for u in range(10):
                    if t_b[0][u][0] != -1:
                        exitnum = 0
                        break
                if exitnum == 1:
                    sys.exit()
                judgemove = 1
                break
    # if fapaitime != 6:
    for i in range(10):
        if t_b[0][i][0] == -1:
            judgemove = 0
            t = -1
            for j in range(10):
                if j == i:
                    continue
                else:
                    if t_b[1][j][1][1] > 170:
                        t = j
                        break
            if t != -1:
                pg.click(x=t_b[1][t][0][0], y=t_b[1][t][0][1])
                sleep(0.1)
                pg.click(x=t_b[1][i][0][0], y=t_b[1][i][0][1])
                sleep(0.5)
                pg.moveTo(100, 100)
                img = pg.screenshot(region=[0, 0, 2559, 1439])  # x,y,w,h
                img.save('example.PNG')
                img = cv2.imread('example.PNG')
                listpos_hang = judgefinish(img)
                inf[i] = number_con(img[listpos_hang[i][0]:listpos_hang[i][1], listpos_lie[i][0]:listpos_lie[i][1]])
                inf[i][1] = inf[i][1] + [photo_x[i], 145]
                inf[t] = number_con(img[listpos_hang[t][0]:listpos_hang[t][1], listpos_lie[t][0]:listpos_lie[t][1]])
                inf[t][1] = inf[t][1] + [photo_x[t], 145]
                t_b = gettb(inf)
                judgemove = 1
            else:
                for j in range(10):
                    if t_b[1][j][0][1] > 170:
                        pg.doubleClick(x=t_b[1][j][0][0], y=t_b[1][j][0][1])
                        sleep(0.1)
                        pg.click(x=t_b[1][i][0][0], y=t_b[1][i][0][1])
                        sleep(0.5)
                        pg.moveTo(100, 100)
                        img = pg.screenshot(region=[0, 0, 2559, 1439])  # x,y,w,h
                        img.save('example.PNG')
                        img = cv2.imread('example.PNG')
                        listpos_hang = judgefinish(img)
                        inf[i] = number_con(
                            img[listpos_hang[i][0]:listpos_hang[i][1], listpos_lie[i][0]:listpos_lie[i][1]])
                        inf[i][1] = inf[i][1] + [photo_x[i], 145]
                        inf[j] = number_con(
                            img[listpos_hang[j][0]:listpos_hang[j][1], listpos_lie[j][0]:listpos_lie[j][1]])
                        inf[j][1] = inf[j][1] + [photo_x[j], 145]
                        t_b = gettb(inf)
                        break
    if judgemove == 0:
        fapai()
        # fapaitime = fapaitime + 1
        sleep(0.5)
        img = pg.screenshot(region=[0, 0, 2559, 1439])  # x,y,w,h
        img.save('example.PNG')
        img = cv2.imread('example.PNG')
        listpos_hang = judgefinish(img)
        for i in range(10):
            inf[i] = number_con(img[listpos_hang[i][0]:listpos_hang[i][1], listpos_lie[i][0]:listpos_lie[i][1]])
            inf[i][1] = inf[i][1] + [photo_x[i], 145]
        t_b = gettb(inf)
    end_time = time()
    run_time = end_time - begin_time
    if run_time > 10:
        break
print('该循环程序运行时间：', run_time)


