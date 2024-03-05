
'''
'''
import os
os.path
import matplotlib.pyplot as plt
import numpy as np
from rlutils.utils import *
from rlutils.vision import *
import pickle
import traceback
from rlutils.plot_utils import config_paper
sys.path.append(os.path.dirname(__file__))

def process_video_moving_cxy(f, mark_frames_save=False):
    '''
    process one video and extract usefult data from a 1 file: red cx,cy
    :param f: path to video
    :param mark_frames_save: save marked frames (red point, blue line)
    returns: episode_cx_arr, episode_cy_arr
    '''
    try:
        frames = read_video_to_frames(f)
        # bx, by, _ = blue_line(frames[1])
        cx_prev=-1#used for debug
        episode_cx_arr, episode_cy_arr, cx_arr_raw = np.zeros(len(frames)), np.zeros(len(frames)), np.zeros(len(frames))
        processed_frames = []

        for i, frame in enumerate(frames):
            cx, cy = red_point(frame, offsetDetection=100)
            cx_arr_raw[i] = cx
            if cx == -1 and cx_prev>10 and cx_prev<400:
                #SAVE frame with cv2
                cv2.imwrite(f[:-4]+f'bug-{i}.jpg',frame)
            cx_prev=cx
            #PROJECT ON BLUE or take cx (COMMENT)
            # cx = np.dot([cx, cy], np.asarray(bx) - np.asarray(by)) / 250000
            episode_cx_arr[i] = cx
            episode_cy_arr[i] = cy
            if mark_frames_save:
                processed_frames.append(mark_frame_red_point_blue_line(frame, cx, cy))

        episode_cx_arr-=episode_cx_arr[0]
        episode_cy_arr-=episode_cy_arr[0]

        if processed_frames!=[]:
            save_video_from_frames(processed_frames, f[:-4] + '_processed.mp4')
    except:
        print('error')
        traceback.print_exc()
        return [], []
    return episode_cx_arr, episode_cy_arr

fig,ax = plt.subplots()
#plot RL from video
rl_path= '/logs/logs_phi_30/152/ep-203-vid-.mp4'
cx_run, cy_arr = process_video_moving_cxy(rl_path)
plt.plot(cx_run)
plt.show()

# cap=cv2.VideoCapture(rl_path)
# frame_rate = cap.get(cv2.CAP_PROP_FPS)
# print(f'frame rate {frame_rate}')
# cx_run-=cx_run[0]
# cx = np.max(cx_run)/len(cx_run)*frame_rate
# # # PLOT RL point learining
# ax.plot(2.8, cx, 'r*', label=r'RL: speed $\overline{\dot{x}}$')