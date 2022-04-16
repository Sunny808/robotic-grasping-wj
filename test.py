import skimage.transform as st
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from utils.dataset_processing.grasp import Grasp

# 构建测试图片
shape = (100, 100)
image = np.zeros(shape)  # 背景图
rr, cc = Grasp((shape[0]/2, shape[1]/2), np.pi/6, 40, 1).as_gr.polygon_coords(shape)
image[rr, cc] = 255
rr, cc = Grasp((shape[0]/4, shape[1]/2), -np.pi/6, 40, 1).as_gr.polygon_coords(shape)
image[rr, cc] = 255
rr, cc = Grasp((shape[0]/3, shape[1]/2), -np.pi/3, 40, 1).as_gr.polygon_coords(shape)
image[rr, cc] = 255
rr, cc = Grasp((shape[0]/2, shape[1]/3), np.pi/3, 40, 1).as_gr.polygon_coords(shape)
image[rr, cc] = 255
rr, cc = Grasp((shape[0]/2, shape[1]/3), np.pi/2, 40, 1).as_gr.polygon_coords(shape)
image[rr, cc] = 255
# idx = np.arange(25, 75)  # 25-74序列
# idx1 = np.arange(26, 76)
# image[idx[::-1], idx] = 255  # 线条\
# image[idx[::-1], idx1] = 255  # 线条\
# image[idx, idx] = 255  # 线条/

# hough线变换
h, theta, d = st.hough_line(image)

# 生成一个一行三列的窗口（可显示三张图片）.
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = (8, 24))
plt.tight_layout()

# 显示原始图片
ax0.imshow(image, plt.cm.gray)
ax0.set_title('Input image')
ax0.set_axis_off()

# 显示hough变换所得数据
ax1.imshow(np.log(1 + h))
ax1.set_title('Hough transform')
ax1.set_xlabel('Angles (degrees)')
ax1.set_ylabel('Distance (pixels)')
ax1.axis('image')

# 显示检测出的线条
ax2.imshow(image, plt.cm.gray)
row1, col1 = image.shape
for _, angle, dist in zip(*st.hough_line_peaks(h, theta, d)):
	y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
	y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
	ax2.plot((0, col1), (y0, y1), '-r')
ax2.axis((0, col1, row1, 0))
ax2.set_title('Detected lines')
ax2.set_axis_off()
plt.show()
