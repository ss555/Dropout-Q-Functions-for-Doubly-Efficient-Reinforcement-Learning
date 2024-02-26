'''
paths is a list of paths to videos
process all videos(extract red point coords) and save bug image if there is one.
'''
import os.path
import sys
import matplotlib.pyplot as plt
import numpy as np
from rlutils.utils import *
from rlutils.vision import *
import pickle
import traceback
from rlutils.plot_utils import config_paper
import cv2
from glob import glob

cos = config_paper()
offsetDetection = 100
kernel = np.ones((5, 5), np.uint8)

def red_point_debug(img, offsetDetection=100,i=0):
    '''
    param:
    return the center of the red point
    '''
    try:
        try:
            if img==[] or len(img.shape) != 3:
                raise Exception('image must have 3 channels')
        except:
            raise Exception('image must have 3 channels')
        # Convert the image from BGR to HSV
        hsv = cv2.cvtColor(img[offsetDetection:-offsetDetection, :], cv2.COLOR_BGR2HSV)

        # Define a range of red color in HSV
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # Combine the two masks
        mask = mask1 + mask2
        # dilate and erode the mask
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        cx, cy = detect_point_from_mask(mask)
        return cx, cy + offsetDetection, mask

    except Exception as e:
        cv2.imwrite(f'./bug_{i}.jpg', img)
        print(e)
        traceback.print_exc()
        print('error with red point')
        raise IOError

#choose img or video imgs(takes 1 incorrect) for debug
paths=[path for path in glob('/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/182/*.mp4')]
# paths=[path for path in glob('/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/152/*.mp4')]
assert paths!=[], 'no videos found'
i=0
for path in paths:
    print(path)
    cap = cv2.VideoCapture(path)
    cx_arr, cy_arr = [], []
    i+=1
    j=0
    while (cap.isOpened()):
        j+=1
        ret, frame = cap.read()
        if ret == False:
            break
        cx, cy, _ = red_point_debug(frame, offsetDetection=offsetDetection, i=j)
        # cx_arr.append(cx)
        # cy_arr.append(cy)
        if cx == -1.0 or cx == 0.0:
            print('red point not found')
            cv2.imwrite(f'./bug_{j}_{i}.jpg', frame)
            sys.exit(0)

# cx_arr = np.array(cx_arr)
# cx_arr = cx_arr - cx_arr[0]
# fig, ax = plt.subplots(1,1, figsize=(5,5))
# plt.plot(cx_arr)
# plt.show()

#y evolution
# fig, ax = plt.subplots(1,1, figsize=(5,5))
# plt.plot(cy_arr)
# plt.show()
sys.exit(0)

cx, cy, mask = red_point_debug(img, offsetDetection=offsetDetection)
print(cx, cy)
def plot_img_mask(img, mask, cx, cy, offsetDetection=100):
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].imshow(img,)
    ax[1].imshow(mask)
    ax[2].imshow(img)
    ax[2].scatter(cx, cy, s=100, c='r', marker='x')
    plt.show()
#show the image, mask and the detected red point
plot_img_mask(img, mask, cx, cy, offsetDetection=offsetDetection)

from rlutils.vision import *

for f in os.listdir('/home/sardor/1-THESE/2-Robotic_Fish/2-DDPG/deepFish/servo-experiment/results/newCamera-fs/1/'):
    if f.endswith('.jpg'):
        img = cv2.imread(
            '/home/sardor/1-THESE/2-Robotic_Fish/2-DDPG/deepFish/servo-experiment/results/newCamera-fs/1/' + f)
        cx, cy = red_point(img)
        if cx == -1:
            cx, cy, mask = red_point_debug(img)
            plot_img_mask(img, mask, cx, cy)
            print(f)

# Apply a morphological transformation to remove small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# Find the contour of the red point
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the list of contours and find the contour with the minimum area that is greater than the minimum surface area limit
pcnts = []
min_area_contour = 50
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area_contour:
        pcnts.append(contour)
    else:
        print(f'contour area {area} is less than {min_area_contour}')

# Get the coordinates of the center of the red point
if len(pcnts) > 0:
    # c = max(contours, key=cv2.contourArea)
    c = max(pcnts, key=cv2.contourArea)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # print("Red point found at ({}, {})".format(cx, cy))
else:
    cx, cy = -1, -1  # print("Red point not found")

erosion = cv2.dilate(mask, kernel, iterations=2)
erosion = cv2.erode(erosion, kernel, iterations=2)
plt.imshow(erosion)
plt.show()
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    print(cv2.contourArea(contour))