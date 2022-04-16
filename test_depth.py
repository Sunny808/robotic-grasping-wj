import cv2.cv2 as cv
import numpy as np
from utils.data.cornell_data import CornellDataset, TestCornellDataset
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from utils.visualisation.plot import plot_results

idx = 88
rot = 0
zoom_factor = 1.0
grasp = CornellDataset('cornell')
test_grasp = TestCornellDataset('cornell')

np.set_printoptions(threshold=np.inf)

# Load the grasps
bbs = grasp.get_gtbb(idx, rot, zoom_factor)  # 这里返回的是GraspRectangle类
depth_img = grasp.get_depth(idx, rot, zoom_factor)

depth = np.array(depth_img)
rgb_img = grasp.get_rgb(idx, normalise = False)# 这里修改normalise为False，才能不转换通道顺序进行输出
rgb = np.array(rgb_img)

cv.imshow('rgb', rgb_img)
cv.imshow('depth', depth)

print(depth)
cv.waitKey(0)
