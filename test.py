import numpy as np

from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt


# 编写一个函数，生成测试图像
def microstructure(l = 256):
	n = 5
	x, y = np.ogrid[0:l, 0:l]
	mask = np.zeros((l, l))
	generator = np.random.RandomState(1)
	points = l * generator.rand(2, n ** 2)
	mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
	mask = ndi.gaussian_filter(mask, sigma = l / (4. * n))
	return mask > mask.mean()


data = microstructure(l = 64)  # 生成测试图像

# 计算中轴和距离变换值
skel, distance = morphology.medial_axis(data, return_distance = True)

# 中轴上的点到背景像素点的距离
dist_on_skel = distance * skel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))
ax1.imshow(data, cmap = plt.cm.gray, interpolation = 'nearest')
# 用光谱色显示中轴
ax2.imshow(dist_on_skel, cmap = plt.cm.spectral, interpolation = 'nearest')
ax2.contour(data, [0.5], colors = 'w')  # 显示轮廓线

fig.tight_layout()
plt.show()
