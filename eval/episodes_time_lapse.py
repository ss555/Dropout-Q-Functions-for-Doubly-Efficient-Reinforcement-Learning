'''
generates time lapse video from video episodes
'''
# importing the necessary libraries
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os
import sys
import glob
from rlutils.vision import *
from matplotlib import pyplot as plt
import pandas as pd

l='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs-fish/152'
color=(0,0,0,255)
font_scale = 1
color = (255, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
time_frames=[]
names=glob(os.path.join(l,'*.mp4'))
idx = [int(n.split('ep-')[-1].split('-vid')[0]) for n in names]

for i, v in enumerate([names[i] for i in np.argsort(idx)]):
    print(v)
    # Opens the Video file
    cap = cv2.VideoCapture(v)
    j = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        # if i%30==0:
        if j%5==0:
            cv2.putText(frame, f'episode-{int(i)}', (50, 50), font, font_scale, color, 2, cv2.LINE_AA)
            time_frames.append(frame)
        j+=1
    cap.release()
    cv2.destroyAllWindows()

time_frames = np.array(time_frames)
save_video_from_frames(time_frames, './time_lapse.mp4', fps=60)