import numpy as np
from cv2 import cv2


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
		if p_img[point] != 0:
			num += 1

	if num > 2:# cross point
		print('find burr:', start_point)
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


def find_burr_point(basic_point, points_set, image):
	r, c = basic_point
	num = 0
	flag = False
	for point in points_set:
		if image[point] != 0:
			num += 1

	if num == 3:
		flag = True
		for point in points_set:
			if image[point] == 0:
				image[r + (r - point[0]), c + (c - point[1])] = 0
				print('clear burr point:', [r + (r - point[0]), c + (c - point[1])])
	elif num == 4:
		flag = True
		for point in points_set:
			x, y = point
			points = [
				(x - 1, y - 1), (x - 1, y), (x - 1, y - 1),
				(x, y - 1),                 (x, y + 1),
				(x + 1, y - 1), (x + 1, y), (x + 1, y - 1)
			]
			q = 0
			for dot in points:
				if image[dot] != 0:
					q += 1
			if q > 3:
				continue
			else:
				image[point] = 0
				print('clear burr point:', point)
	return flag


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


def remove_burr_points(basic_point_set, image):
	if len(basic_point_set) == 1:
		image[basic_point_set] = 0
		print('delete burr point:', basic_point_set)
	else:
		for basic_point in basic_point_set:
			x, y = basic_point
			points = [
				(x - 1, y - 1), (x - 1, y), (x - 1, y - 1),
				(x, y - 1), (x, y + 1),
				(x + 1, y - 1), (x + 1, y), (x + 1, y - 1)
			]
			for point in points:
				if image[point] != 0:
					r, c = point
					points_cross = [(r-1, c), (r+1, c), (r, c+1), (r, c-1)]
					points_cross_cline = [(r-1, c-1), (r+1, c+1), (r-1, c+1), (r+1, c-1)]

					if find_burr_point(point, points_cross, image):
						break
					else:
						find_burr_point(point, points_cross_cline, image)


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
					if image[point] != 0:
						flag += 1

				if flag == 1:
					bound = [(r, c)]
					print('find start point[{}, {}]'.format(r, c))
					index_img[r, c] = 1
					bounds.append(bound)
					burr(bounds_cross, bounds, image, index_img)

	length = len(bounds_cross)
	print('the length of cross line is', length)
	num = length
	cross_point = []
	flag = []

	list.sort(bounds_cross, key = lambda k: len(k), reverse = False)

	max_length = 0
	for cross_set in bounds_cross:
		if len(cross_set) > max_length:
			max_length = len(cross_set)
		cross_point.append(cross_set[-1])
		flag.append(True)
	print('cross set is:', cross_point)

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
						print('delete cross point:', cross_point[i])
						num -= 1
						del_set = bounds_cross[i]
						for point in del_set[0:-1]:
							image[point] = 0
							# print('delete point:', point)
						if is_burr_point(del_set[-1], image):
							image[del_set[-1]] = 0
							delete_burr_point(del_set[-1], image)
						flag[j] = False
						flag[i] = False

		for i in range(length - 1):
			for j in range(i + 1, length):
				if flag[i]:
					if is_repeat(cross_point[i], cross_point[j]):
						print('delete repeat point {}.'.format(bounds_cross[i][-1]))
						del_set = bounds_cross[i]
						for point in del_set:
							image[point] = 0
						if is_burr_point(del_set[-1], image):
							delete_burr_point(del_set[-1], image)
						flag[i] = False
						flag[j] = False
						num -= 1

		for i in range(length-1):
			if flag[i]:
				print('Delete the spare line {}.'.format(bounds_cross[i][-1]))
				del_set = bounds_cross[i]
				for point in del_set[0:-1]:
					image[point] = 0
				if is_burr_point(del_set[-1], image):
					image[del_set[-1]] = 0
					delete_burr_point(del_set[-1], image)
				flag[i] = False
				num -= 1

	print('num is', num)
	return image, num


img = cv2.imread('thin.png', 0)
img1 = np.array(img)
img1, n = delete_burr_line(img1)
cv2.namedWindow('thin0', 0)
cv2.resizeWindow('thin0', 800, 800)
cv2.imshow('thin0', img1)
# cv2.imwrite('thin.png', img1)
cv2.waitKey(0)
