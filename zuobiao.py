import pyautogui as pg
import time
import cv2
import numpy as np
from time import *

# sleep(2)
# begin_time = time()
# while(1):
#     pg.click(x=2447, y=1316)
#     end_time = time()
#     run_time = end_time - begin_time
#     if run_time > 10:
#         break

while True:
    x,y = pg.position()
    print("\r",(x,y),end="")
    sleep(0.5)