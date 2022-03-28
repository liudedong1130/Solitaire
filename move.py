import numpy as np
import pyautogui as pg
from time import *


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
    return t_b_number,t_b_position

def fapai():
    pg.click(x=2003, y=1126)
    sleep(1)

