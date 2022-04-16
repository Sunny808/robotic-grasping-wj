import glob
import os
import numpy as np
from cv2 import cv2

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
		image = cv2.line(image, points[index], points[(index+1) % 4], color)
	return image


idx = 2
rgb_img = cv2.imread(rgb_files[idx])
point_poss = load_from_cornell_file(grasp_files_pos[idx])
point_negs = load_from_cornell_file(grasp_files_neg[idx])
color_sets = [[255, 0, 0], [0, 255, 0]]
for point_pos in point_poss:
	rgb_img = draw_line(rgb_img, point_pos, color_sets)
color_sets = [[255, 255, 0], [0, 0, 255]]
for point_neg in point_negs:
	rgb_img = draw_line(rgb_img, point_neg, color_sets)

cv2.imwrite('cornell/RectangleLable/01.png', rgb_img)



