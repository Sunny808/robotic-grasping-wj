from cv2 import cv2 as cv
import numpy as np
from utils.data.cornell_data import CornellDataset, TestCornellDataset
import matplotlib.pyplot as plt
from utils.dataset_processing import grasp
import glob
import os

grasp_files_pos = glob.glob(os.path.join('cornell', '*', 'pcd*cpos.txt'))
grasp_files_pos.sort()
rgb_files = [f.replace('cpos.txt', 'r.png') for f in grasp_files_pos]
grasp_files_neg = [f.replace('pos.txt', 'neg.txt') for f in grasp_files_pos]


def _gr_text_to_no(l, offset = (0, 0)):
	"""
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
	x, y = l.split()
	return [int(round(float(x))) - offset[0], int(round(float(y))) - offset[1]]


def load_from_cornell_file(fname):
	"""
	Load grasp rectangles from a Cornell dataset grasp file.
	:param fname: Path to text file.
	:return: GraspRectangles()
	"""
	grs = []
	with open(fname) as f:
		while True:
			# Load 4 lines at a time, corners of bounding box.
			p0 = f.readline()
			if not p0:
				break  # EOF
			p1, p2, p3 = f.readline(), f.readline(), f.readline()
			try:
				gr = np.array([
					_gr_text_to_no(p0),
					_gr_text_to_no(p1),
					_gr_text_to_no(p2),
					_gr_text_to_no(p3)  # 4X2array
				])

				grs.append(gr)

			except ValueError:
				# Some files contain weird values.
				continue
	return grs


def draw_line(image, points, colors):
	for index in range(4):
		if (index % 2) == 0:
			color = colors[0]
		else:
			color = colors[1]
		image = cv.line(image, points[index], points[(index+1) % 4], color)
	return image


def crop_image(image, bb):
	center = bb.center  # return 1X2 array
	left = max(0, min(center[1] - 224 // 2, 640 - 224))
	top = max(0, min(center[0] - 224 // 2, 480 - 224))
	bottom = min(480, top + 224)
	right = min(640, left + 224)
	# print(center, top, left, bottom, right)
	img = image[top:bottom, left:right]
	return img


for idx in range(492, 500):
	print(idx)
	rot = 0
	zoom_factor = 1.0
	grasping = CornellDataset('cornell')
	test_grasp = TestCornellDataset('cornell')

	# Load the grasps
	bbs = grasping.get_gtbb(idx, rot, zoom_factor)  # 这里返回的是GraspRectangle类
	gtbbs = grasp.GraspRectangles.load_from_cornell_file(grasp_files_pos[idx])
	# for bb in bbs:
	#    print(bb.angle, bb.width)
	rgb_img = cv.imread(rgb_files[idx])
	point_poss = load_from_cornell_file(grasp_files_pos[idx])
	point_negs = load_from_cornell_file(grasp_files_neg[idx])
	color_sets = [[255, 0, 0], [0, 0, 255]]
	for point_pos in point_poss:
		rgb_img = draw_line(rgb_img, point_pos, color_sets)
	color_sets = [[0, 255, 0], [255, 255, 0]]
	for point_neg in point_negs:
		rgb_img = draw_line(rgb_img, point_neg, color_sets)
	rgb_img = crop_image(rgb_img, gtbbs)
	raw_img = crop_image(cv.imread(rgb_files[idx]), gtbbs)
	cv.namedWindow('rgb{}'.format(idx), cv.WINDOW_NORMAL)
	cv.imshow('rgb{}'.format(idx), rgb_img)

	pos_img, ang_img, width_img = bbs.draw((224, 224))
	# cv.namedWindow('pos_img', cv.WINDOW_NORMAL)
	# cv.imshow('pos_img', pos_img * 255)
	gs = grasp.detect_grasps(pos_img, ang_img, width_img=width_img, no_grasps=1)

	# Load the test_grasps
	test_bbs = test_grasp.get_gtbb(idx, rot, zoom_factor)  # 这里返回的是GraspRectangle类
	# for bb in test_bbs:
	#    print(bb.angle, bb.width)
	rgb_img1 = test_grasp.get_rgb(idx, normalise = False)  # 这里修改normalise为False，才能不转换通道顺序进行输出
	rgb1 = np.array(rgb_img1)

	depth_img1 = test_grasp.get_depth(idx, rot, zoom_factor)
	depth1 = np.array(depth_img1)

	pos_img1, ang_img1, width_img1 = test_bbs.draw((224, 224))

	fig = plt.figure(figsize = (10, 10))

	ax = fig.add_subplot(3, 2, 1)
	plot = ax.imshow(pos_img, cmap='jet', vmin=0, vmax= 1)
	ax.set_title('DoubleLinear_img')
	ax.axis('off')
	plt.colorbar(plot)

	ax = fig.add_subplot(3, 2, 2)
	plot = ax.imshow(pos_img1, cmap = 'jet', vmin = 0, vmax = 1)
	ax.set_title('RawPosition')
	ax.axis('off')
	plt.colorbar(plot)

	ax = fig.add_subplot(3, 2, 3)
	ax.set_title('Width')
	plot = ax.imshow(width_img, cmap = 'jet', vmin = 0, vmax = 100)
	ax.axis('off')
	plt.colorbar(plot)

	ax = fig.add_subplot(3, 2, 4)
	ax.set_title('RawWidth')
	plot = ax.imshow(width_img1, cmap = 'jet', vmin = 0, vmax = 100)
	ax.axis('off')
	plt.colorbar(plot)

	ax = fig.add_subplot(3, 2, 5)
	plot = ax.imshow(ang_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
	ax.set_title('Angle')
	ax.axis('off')
	plt.colorbar(plot)

	ax = fig.add_subplot(3, 2, 6)
	plot = ax.imshow(ang_img1, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
	ax.set_title('RawAngle')
	ax.axis('off')
	plt.colorbar(plot)

	plt.pause(0.1)
	fig.canvas.draw()
	cv.waitKey(0)

