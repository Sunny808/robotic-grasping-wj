import numpy as np
from cv2 import cv2
from skimage import morphology
from skimage.filters import gaussian


def burr(bounds_cross, bounds, p_img, index_img):
	start_point = tuple(bounds[-1][-1])
	r, c = start_point
	points = [
		(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
		(r, c - 1), (r, c + 1),
		(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
	]

	num = 0

	for point in points:
		if point[0] >= p_img.shape[1] or point[1] >= p_img.shape[0]:
			continue
		if p_img[point] != 0:
			num += 1

	if num > 2:# cross point
		index_img[start_point] = 0
		bounds_cross.append(bounds[-1])
		return 0

	for point in points:
		if p_img[point] != 0 and index_img[point] == 0:
			bounds[-1].append(point)
			index_img[point] = 1
			burr(bounds_cross, bounds, p_img, index_img)


def is_repeat(point_index, point):
	cross_point = []
	x, y = point_index
	point_index_set = [
		(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
		(x, y - 1),                 (x, y + 1),
		(x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
	]

	r, c = point
	point_set = [
		(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
		(r, c - 1),                 (r, c + 1),
		(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
	]

	for dot in point_index_set:
		for spot in point_set:
			if dot == spot:
				cross_point.append(dot)
	return cross_point


def is_burr_point(burr_point, image):
	flag = False
	r, c = burr_point
	dots = [
		(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
		(r, c - 1), (r, c + 1),
		(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
	]
	no = 0
	p = []
	for dot in dots:
		if dot[0] >= image.shape[1] or dot[1] >= image.shape[0]:
			continue
		if image[dot] != 0:
			p.append(dot)
			no += 1
	if no > 2:
		flag = True
	elif no == 2:
		if abs(p[0][0] - p[1][0]) + abs(p[0][1] - p[1][1]) == 1:
			flag = True

	return flag


def delete_burr_point(burr_point, image):
	r, c = burr_point
	points = [
		(r - 1, c),
		(r, c - 1), (r, c + 1),
		(r + 1, c)
	]
	for point in points:
		if point[0] >= image.shape[1] or point[1] >= image.shape[0]:
			continue
		if image[point]:
			row, column = point
			ps = [
				(row - 1, column),
				(row, column - 1), (row, column + 1),
				(row + 1, column)
			]
			count = 0
			p_set = []
			for p in ps:
				if image[p]:
					count += 1
					p_set.append(p)
			if count > 2:
				image[point] = 0
			elif count == 2:
				if p_set[0][0] != p_set[1][0] and p_set[0][1] != p_set[1][1]:
					image[point] = 0


def delete_burr_line(image):
	bounds = []
	bounds_cross = []
	index_img = np.zeros((image.shape[1], image.shape[0]))
	for r in range(1, image.shape[1] - 1):
		for c in range(1, image.shape[0] - 1):
			if index_img[r, c] == 0 and image[r, c] != 0:
				flag = 0
				points = [
					(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
					(r, c - 1), (r, c + 1),
					(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
				]

				for point in points:
					if point[0] >= image.shape[1] or point[1] >= image.shape[0]:
						continue
					if image[point] != 0:
						flag += 1

				if flag == 1:
					bound = [(r, c)]
					index_img[r, c] = 1
					bounds.append(bound)
					burr(bounds_cross, bounds, image, index_img)

	length = len(bounds_cross)
	num = length
	cross_point = []
	flag = []

	list.sort(bounds_cross, key = lambda n: len(n), reverse = False)

	max_length = 0
	for cross_set in bounds_cross:
		if len(cross_set) > max_length:
			max_length = len(cross_set)
		cross_point.append(cross_set[-1])
		flag.append(True)

	if length == 1:# Q
		if len(bounds_cross[-1]) < 10:
			del_set = bounds_cross[-1]
			for point in del_set[0:-1]:
				image[point] = 0

			if is_burr_point(del_set[-1], image):
				image[del_set[-1]] = 0
			num = 0
	else:
		for i in range(length-1):
			for j in range(i+1, length):
				if flag[i]:
					if cross_point[i] == cross_point[j]:
						num -= 1
						del_set = bounds_cross[i]
						for point in del_set[0:-1]:
							image[point] = 0
						if is_burr_point(del_set[-1], image):
							image[del_set[-1]] = 0
							delete_burr_point(del_set[-1], image)
						flag[i] = False
						flag[j] = False

		for i in range(length - 1):
			for j in range(i + 1, length):
				if flag[i]:
					if is_repeat(cross_point[i], cross_point[j]):
						del_set = bounds_cross[i]
						for point in del_set:
							image[point] = 0
						if is_burr_point(del_set[-1], image):
							delete_burr_point(del_set[-1], image)
						flag[i] = False
						flag[j] = False
						num -= 1

		for i in range(length - 1):
			if flag[i]:
				del_set = bounds_cross[i]
				for point in del_set[0:-1]:
					image[point] = 0
				if is_burr_point(del_set[-1], image):
					image[del_set[-1]] = 0
					delete_burr_point(del_set[-1], image)
				flag[i] = False
				num -= 1

	return image, num


def find_line_segment(bounds, p_img, index_img, a_img):
	start_point = tuple(bounds[-1][-1])
	r, c = start_point

	points = [
		(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
		(r, c - 1), (r, c + 1),
		(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
	]

	for point in points:
		if point[0] >= p_img.shape[1] or point[1] >= p_img.shape[0]:
			continue
		if p_img[point]:
			if index_img[point] > 0:
				continue
			if abs((a_img[point] - a_img[start_point] + np.pi / 2) % np.pi - np.pi / 2) <= np.pi/6:
				bounds[-1].append(point)
				index_img[point] = 1
				find_line_segment(bounds, p_img, index_img, a_img)
			else:
				print('angle out of range.')
				bounds.append([point])# Create a new point.
				index_img[point] = 1
				find_line_segment(bounds, p_img, index_img, a_img)


def linear_interpolation(point_sets, weight_img, index_img):
	bound_set = []
	for point_set in point_sets:
		length = len(point_set)
		xp = [0, length // 2, length - 1]
		fp = [0.5, 1, 0.5]
		a = list(range(length))
		linear_value = np.interp(a, xp, fp)
		for num, (r, c) in enumerate(point_set):
			weight_img[r, c] = linear_value[num]
			index_img[r, c] = 3
			bound_set.append((r, c))
	return bound_set, weight_img, index_img


def find_bound(image, a_img):
	bounds = []
	bound_set = []
	index_img = np.zeros((image.shape[1], image.shape[0]))

	image, length = delete_burr_line(image)
	if length > 1:
		for i in range(length):
			image, num = delete_burr_line(image)
			if num <= 2:
				break

	# skeleton_img = image
	# cv2.namedWindow('skeleton_pos', cv2.WINDOW_NORMAL)
	# cv2.imshow('skeleton_pos', skeleton_img * 255)

	for r in range(1, image.shape[1] - 1):
		for c in range(1, image.shape[0] - 1):
			if index_img[r, c] == 0 and image[r, c] != 0:
				flag = 0
				points = [
					(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
					(r, c - 1), (r, c + 1),
					(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
				]
				for point in points:
					if point[0] >= image.shape[1] or point[1] >= image.shape[0]:
						continue
					if image[point] != 0:
						flag += 1

				if flag == 1:# Find start point.
					bound = [(r, c)]
					index_img[r, c] = 1
					bounds.append(bound)
					find_line_segment(bounds, image, index_img, a_img)

	weight_img = np.zeros((224, 224))
	index_img = np.zeros((224, 224))

	if bounds:
		bound_set, weight_img, index_img = linear_interpolation(bounds, weight_img, index_img)

	# The line include a loop.
	for r in range(1, image.shape[1] - 1):
		for c in range(1, image.shape[0] - 1):
			if image[r, c] and index_img[r, c] == 0:
				bound_set.append((r, c))
				weight_img[r, c] = 1.0
				index_img[r, c] = 3

	return bound_set, weight_img, index_img


def broadcast_bound(bound_set, weight_img, index_img, width_out, iteration = 10, offset = 0.1):
	temp = []
	for row, column in bound_set:
		if weight_img[row, column] <= offset and width_out[row, column] == 0:
			continue
		if row < (weight_img.shape[1] - 1) and column < (weight_img.shape[0] - 1):
			points = {
				(row - 1, column - 1): 1, (row - 1, column): 2, (row - 1, column + 1): 1,
				(row, column - 1): 2, (row, column + 1): 2,
				(row + 1, column - 1): 1, (row + 1, column): 2, (row + 1, column + 1): 1
			}
			for point, mark in points.items():
				if point[0] >= weight_img.shape[1] or point[1] >= weight_img.shape[0]:
					continue
				if index_img[point] < 2:
					weight_img[point] = weight_img[row, column] - offset
					index_img[point] = mark

	for r in range(weight_img.shape[0]):
		for c in range(weight_img.shape[1]):
			if index_img[r, c] == 2 or index_img[r, c] == 1:
				index_img[r, c] = 3
				temp.append((r, c))

	if iteration > 0:
		return broadcast_bound(temp, weight_img, index_img, width_out, iteration - 1)
	else:
		return weight_img


def gr_gaussian(pos_out, ang_out, width_out):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	pos_out = cv2.morphologyEx(pos_out, cv2.MORPH_CLOSE, kernel)
	pos_out = cv2.morphologyEx(pos_out, cv2.MORPH_OPEN, kernel)
	width_out = cv2.morphologyEx(width_out, cv2.MORPH_CLOSE, kernel)
	width_out = cv2.morphologyEx(width_out, cv2.MORPH_OPEN, kernel)
	ang_out = cv2.morphologyEx(ang_out, cv2.MORPH_CLOSE, kernel)
	ang_out = cv2.morphologyEx(ang_out, cv2.MORPH_OPEN, kernel)
	cv2.imshow('pos_out', pos_out * 255)
	cv2.imwrite('pos.png', pos_out)

	thin0 = morphology.thin(pos_out)
	thin = thin0.astype(np.uint8)
	# threshold = 1
	# skeleton, distance = morphology.medial_axis(pos_out, return_distance = True)
	# skeleton = skeleton.astype(np.float32)
	# dist_on_skeleton = distance * skeleton
	# dist_on_skeleton[dist_on_skeleton <= threshold] = 0

	bound_point, pos_out, matrix_flag = find_bound(thin, ang_out)
	pos_out = broadcast_bound(bound_point, pos_out, matrix_flag, width_out)
	
	# cv2.namedWindow('thin', cv2.WINDOW_NORMAL)
	# cv2.imshow('thin', thin * 255)
	# cv2.imwrite('thin.png', thin * 255)
	# cv2.imshow('pos_out2', pos_out * 255)
	# cv2.waitKey(0)
	
	pos_out = gaussian(pos_out, 2.0, preserve_range = True)
	max_pos = np.max(pos_out)
	if max_pos:
		pos_out /= max_pos

	return pos_out, ang_out, width_out
