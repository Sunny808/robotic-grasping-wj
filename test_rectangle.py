import cv2.cv2 as cv
import numpy as np
from utils.data.cornell_data import CornellDataset
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from skimage import morphology


idx = 73
rot = 0
zoom_factor = 1.0
grasp = CornellDataset('cornell')

# Load the grasps
bbs = grasp.get_gtbb(idx, rot, zoom_factor)  # 这里返回的是GraspRectangle类
# for bb in bbs:
#    print(bb.angle, bb.width)
rgb_img = grasp.get_rgb(idx, normalise = False)  # 这里修改normalise为False，才能不转换通道顺序进行输出
rgb = np.array(rgb_img)
depth_img = grasp.get_depth(idx, rot, zoom_factor)
depth = np.array(depth_img)

pos_img, ang_img, width_img = bbs.draw((224, 224))
cv.imshow('rgb', rgb_img)
cv.imshow('pos', pos_img)
cv.imshow('ang', ang_img)
cv.imshow('width', width_img)

thin0 = morphology.thin(pos_img)  # 骨架提取
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
thin = thin0.astype(np.uint8) * 255
# thin = cv.morphologyEx(thin, cv.MORPH_CLOSE, kernel, iterations=1)
# cv.namedWindow('thin', cv.WINDOW_NORMAL)
cv.imshow('thin', thin)


def find_line_segment(bounds, p_img, index_img, a_img):
    start_point = tuple(bounds[-1][-1])
    r, c = start_point

    points = [
                (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
                (r, c - 1),                        (r, c + 1),
                (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
            ]

    num = 0

    for point in points:# Delete the burr
        if p_img[point] != 0:
            num += 1
    if num > 2:
        if len(bounds[-1]) < 11:
            for dot in bounds[-1][0:-2]:
                p_img[dot] = 0
                index_img[dot] = 0
            if bounds:
                bounds.pop()
            # print('pop burr')
            return 0

    for point in points:
        num = 0
        if p_img[point] != 0:
            if index_img[point] > 0:
                continue

            if abs(a_img[point] - a_img[start_point]) <= np.pi/6 or \
                    abs(a_img[point] - a_img[start_point]) >= np.pi*5/6:
                num += 1
                bounds[-1].append(point)
                index_img[point] = 1
                find_line_segment(bounds, p_img, index_img, a_img)
            else:
                # print('angle range.')
                # print(a_img[point])
                bounds.append([point])
                index_img[point] = 1
                find_line_segment(bounds, p_img, index_img, a_img)

    if num == 0:
        return 0


def linear_interpolation(point_sets, weight_img, index_img):
    bound_set = []
    print('bound_length:', len(point_sets))
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
    index_img = np.zeros((image.shape[1], image.shape[0]))
    for r in range(1, image.shape[1] - 1):
        for c in range(1, image.shape[0] - 1):
            bound = []
            if index_img[r, c] == 0 and image[r, c] != 0:
                flag = 0
                points = [
                    (r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
                    (r, c - 1),                 (r, c + 1),
                    (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)
                ]
                for point in points:
                    if image[point] != 0:
                        flag += 1

                if flag == 1:
                    bound = [(r, c)]
                    index_img[r, c] = 1
                    bounds.append(bound)
                    find_line_segment(bounds, image, index_img, a_img)

    weight_img = np.zeros((224, 224))
    index_img = np.zeros((224, 224))

    if bounds:
        return linear_interpolation(bounds, weight_img, index_img)

    else:# The line is a loop.
        bound_set = []
        for r in range(1, image.shape[1]):
            for c in range(1, image.shape[0]):
                if image[r, c] != 0:
                    bound_set.append((r, c))
                    weight_img[r, c] = 1.0
                    index_img[r, c] = 3
    return bound_set, weight_img, index_img


def broadcast_bound(bound_set, weight_img, index_img, iteration = 5, offset = 0.1):
    temp = []
    for row, column in bound_set:
        points = {
            (row-1, column-1): 1,   (row-1, column): 2, (row-1, column+1): 1,
            (row, column-1): 2,                         (row, column + 1): 2,
            (row+1, column-1): 1,   (row+1, column): 2, (row+1, column+1): 1
        }

        for point, mark in points.items():
            if index_img[point] < 2:
                weight_img[point] = weight_img[row, column] - offset
                index_img[point] = mark

    for r in range(weight_img.shape[0]):
        for c in range(weight_img.shape[1]):
            if index_img[r, c] == 2 or index_img[r, c] == 1:
                index_img[r, c] = 3
                temp.append((r, c))

    if iteration > 0:
        return broadcast_bound(temp, weight_img, index_img, iteration - 1)
    else:
        return weight_img


"""
def broadcast_bound_cross(bound_set, weight_img, flag_img, iteration = 5, offset = 0.1):
    temp = []
    for row, column in bound_set:
        points = [(row-1, column), (row, column-1), (row, column + 1), (row+1, column)]
        for point in points:
            if flag_img[point] < 2:
                weight_img[point] = weight_img[row, column] - offset
            flag_img[point] += 1

    for r_kernel in range(weight_img.shape[0]):
        for c_kernel in range(weight_img.shape[1]):
            if flag_img[r_kernel, c_kernel] == 2 or flag_img[r_kernel, c_kernel] == 1:
                flag_img[r_kernel, c_kernel] += 1
                temp.append((r_kernel, c_kernel))

    if iteration > 0:
        return broadcast_bound(temp, weight_img, flag_img, iteration-1)
    else:
        return 0
"""

# bound_point, quality_img, matrix_flag = find_bound(thin, ang_img)
# q_img = broadcast_bound(bound_point, quality_img, matrix_flag)
# cv.imshow('pos_thin', q_img)

# q_img = gaussian(q_img, 2.0, preserve_range = True)
ang_img = gaussian(ang_img, 2.0, preserve_range = True)
width_img = gaussian(width_img, 1.0, preserve_range = True)


fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(2, 1, 1)
ax.imshow(rgb_img)
ax.set_title('RGB{}'.format(idx))
ax.axis('off')

# print('max:', np.max(q_img1))
# m = np.max(q_img)
# q_img /= m

ax = fig.add_subplot(2, 1, 2)
plot = ax.imshow(thin, cmap='jet', vmin=0, vmax=1)
ax.set_title('Thin_img')
ax.axis('off')
plt.colorbar(plot)
plt.pause(0.1)
fig.canvas.draw()

'''
fig = plt.figure(figsize = (10, 10))
plot_results(fig = fig,
             rgb_img = rgb_img,
             grasp_q_img = q_img,
             grasp_angle_img = ang_img,
             depth_img = depth_img,
             no_grasps = 1,
             grasp_width_img = width_img)
fig.savefig('img_result.pdf')

fig1 = plt.figure(figsize = (10, 10))
plot_results(fig = fig1,
             rgb_img = rgb_img1,
             grasp_q_img = q_img1,
             grasp_angle_img = ang_img1,
             depth_img = depth_img1,
             no_grasps = 1,
             grasp_width_img = width_img1)
fig1.savefig('img_result_test.pdf')
'''
cv.waitKey(0)
