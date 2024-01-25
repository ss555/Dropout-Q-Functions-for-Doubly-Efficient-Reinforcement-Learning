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
import