import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max
from cv2 import cv2
from .test_gaussian import gr_gaussian


def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs# 这里grs是GraspRectangle类
        else:
            self.grs = []
        self.grs_max = []
        self.grs_min = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
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
                        _gr_text_to_no(p3)# 4X2array
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def integrate_feature(self):
        # length = len(self.grs)
        # grs = []
        # flag = []
        # for i in range(length):
        #     flag.append(True)
        average_grasp_width = 0
        sum_grs = 0

        for index, gr in enumerate(self.grs):  # class and filtrate
            sum_grs += gr.length
        if len(self.grs):
            average_grasp_width = sum_grs/len(self.grs)

        #     if flag[index] and index != length-1:
        #         for i in range(index + 1, length):
        #             gr_next = self.grs[i]
        #             if flag[i]:
        #                 if gr.is_center_cross(gr_next, arrange = 9) and\
        #                         gr.iou_cross(gr_next, angle_threshold = np.pi/6, iou_threshold = 0.25):
        #                     print('delete center_cross and iou_0.25 cross rectangle.')
        #                     if gr.length >= gr_next.length:
        #                         flag[i] = False
        #                     else:
        #                         flag[index] = False
        #
        # for i in range(length):
        #     if ~flag[i]:
        #         self.grs[i].angle = None
        #     grs.append(self.grs[i])
        # self.grs = grs

        threshold = average_grasp_width / 2

        for gr in self.grs:
            if gr.length > threshold:
                self.grs_max.append(gr)
            else:
                self.grs_min.append(gr)

        if len(self.grs_max) >= 3:# remove a cross rectangle.
            index = []
            for i, gr in enumerate(self.grs_max):
                count = 0
                for gr_compare in self.grs_max:
                    if gr.iou_cross(gr_compare, angle_threshold = np.pi/4, iou_threshold = 0.1):
                        count += 1

                if count == len(self.grs_max)/2 + 1:
                    index.append(i)
            if index:
                print('Remove wired cross rectangles.')
                for i in index:
                    del self.grs_max[i]
                    break

        return average_grasp_width

    @staticmethod
    def gr_dilate(grs, pos_out, ang_out, width_out, shape, w_threshold, kernel_size=3):
        """
        Dilate a GraspRectangle according to it's angel
        :param grs:
        :param pos_out: pos
        :param ang_out: ang
        :param width_out: width
        :param kernel_size: the size of the kernel
        :param shape: the size of the image
        :param w_threshold:
        """
        a = np.uint8(np.zeros((kernel_size, kernel_size)))
        str_element_horizon = a.copy()
        str_element_horizon[kernel_size // 2, :] = 1

        _rr = list(range(kernel_size))
        _cc = list(reversed(_rr))

        str_element_diagonal_pos = a.copy()
        str_element_diagonal_pos[_rr, _cc] = 1

        str_element_diagonal_neg = a.copy()
        str_element_diagonal_neg[_rr, _rr] = 1

        str_element_vertical = a.copy()
        str_element_vertical[:, kernel_size // 2] = 1

        gr_previous = None
        for gr in grs:
            pos_index = np.zeros(shape)
            rr, cc = gr.compact_polygon_coords(ratio_length = 3, ratio_width = 1, shape = shape)
            pos_index[rr, cc] = 1.0

            if -np.pi / 8 <= gr.angle < np.pi / 8:
                kernel = str_element_vertical
            elif np.pi / 8 <= gr.angle < 3 * np.pi / 8:
                kernel = str_element_diagonal_neg
            elif -3 * np.pi / 8 <= gr.angle <= -np.pi / 8:
                kernel = str_element_diagonal_pos
            else:
                kernel = str_element_horizon

            pos_index = cv2.dilate(pos_index, kernel)
            pos_coords = np.nonzero(pos_index)

            if gr_previous is not None:
                if gr.iou_cross(gr_previous, angle_threshold = np.pi/6, iou_threshold = 0.25):
                    # width_out[pos_coords] = (gr.length + gr_previous.length)/2
                    # ang_out[pos_coords] = 0
                    print('Is iou cross. The iou is {}'.format(gr.iou_cross(gr_previous)))
                else:
                    pos_out[pos_coords] = 1.0
                    for r, c in np.transpose([pos_coords]):
                        if width_out[r, c]:
                            width_out[r, c] = (gr.length + width_out[r, c])/2
                        else:
                            width_out[r, c] = gr.length
                        ang_out[r, c] = gr.angle
            else:
                pos_out[pos_coords] = 1.0
                width_out[pos_coords] = gr.length
                ang_out[pos_coords] = gr.angle

            gr_previous = gr

        return gr_gaussian(pos_out, ang_out, width_out)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        average_grasp_width = self.integrate_feature()
        if average_grasp_width > 60:
            k_size = 5
        else:
            k_size = 3

        pos_out, ang_out, width_out = self.gr_dilate(self.grs_max, pos_out, ang_out, width_out, shape, k_size)
        if self.grs_min:
            k_size = 3
            print('Feature split.')
            if position:
                pos_out_min = np.zeros(shape)
            else:
                pos_out_min = None
            if angle:
                ang_out_min = np.zeros(shape)
            else:
                ang_out_min = None
            if width:
                width_out_min = np.zeros(shape)
            else:
                width_out_min = None
            pos_out_min, ang_out_min, width_out_min =\
                self.gr_dilate(self.grs_min, pos_out_min, ang_out_min, width_out_min, shape, k_size)

            for r in range(shape[0]):
                for c in range(shape[1]):
                    if pos_out_min[r, c] and pos_out[r, c] < 0.5:
                        pos_out[r, c] = pos_out_min[r, c]
                        ang_out[r, c] = ang_out_min[r, c]
                        width_out[r, c] = width_out_min[r, c]

        for r in range(shape[0]):
            for c in range(shape[1]):
                if pos_out[r, c] <= 0 or width_out[r, c] <= 0:
                    pos_out[r, c] = 0
                    width_out[r, c] = 0
                    ang_out[r, c] = 0

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int) # return 1X2 array


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, ratio_length = 3, ratio_width = 1, shape=None):
        """
        :param shape: Output shape
        :param ratio_length:
        :param ratio_width:
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle,
                     self.length / ratio_length, self.width / ratio_width).as_gr.polygon_coords(shape)
        # 这里self为GraspRectangle类而不是Grasp类

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def iou_cross(self, gr, angle_threshold=np.pi / 6, iou_threshold = 0.25):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :param iou_threshold:
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) <= angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        iou = intersection / union
        if iou > iou_threshold:
            return iou
        else:
            return 0

    def is_center_cross(self, gr, arrange = 9):
        [x, y] = self.center
        [x1, y1] = gr.center
        if abs(x - x1) <= arrange and abs(y - y1) <= arrange:
            return True
        else:
            return False

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom center (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo# y是水平方向，x是垂直方向
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],# 矩形某个角的一个点是由水平和垂直方向的旋转共同决定
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        ).astype(np.float))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * self.angle * 180 / np.pi, self.length * scale,
            self.width * scale)


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)

    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2  # default = 2

        grasps.append(g)

    return grasps
