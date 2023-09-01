"""This is the chessboard-position-search (CPS) module."""


import collections
import itertools
import math

import cv2
import matplotlib.path
import numpy as np
import pyclipper
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from lc2fen.detectboard import debug


def __order_points(pts: list[list]) -> list[list[int]]:
    """Order the four points in the order of TR, TR, BR, and BL.

    This function orders the four points in the order of top left, top
    right, bottom right, and bottom left.

    :param pts: List of four 2D points

    :return: Ordered list of the four 2D points.
    """
    pts = np.float32(pts)
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas the
    # bottom-right point will have the largest sum
    _sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(_sum)]
    rect[2] = pts[np.argmax(_sum)]
    # now, compute the difference between the points, the top-right
    # point will have the smallest difference, whereas the bottom-left
    # will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return __normalize(rect)


def __normalize(points: list[list]) -> list[list[int]]:
    """Normalize the input points."""
    return [[int(a), int(b)] for a, b in points]


def __check_correctness(points: list[list], shape: list):
    """Check that the points are in the given shape."""
    __points = []
    for point in points:
        if 0 <= point[0] <= shape[1] and 0 <= point[1] <= shape[0]:
            __points.append(point)
    return __points


def __remove_duplicates(input_list: list) -> list:
    """Remove duplicates from list containing unhashable elements.

    This function removes duplicate elements from the input list
    containing unhashable elements while preserving order.
    """
    indices = sorted(range(len(input_list)), key=input_list.__getitem__)
    indices = set(
        next(it)
        for k, it in itertools.groupby(indices, key=input_list.__getitem__)
    )
    return [x for i, x in enumerate(input_list) if i in indices]


def __sort_points(pts: list[list]) -> list[list]:
    """Sort points clockwise."""
    mlat = sum(x[0] for x in pts) / len(pts)
    mlng = sum(x[1] for x in pts) / len(pts)

    def __sort(x):
        return (math.atan2(x[0] - mlat, x[1] - mlng) + 2 * math.pi) % (
            2 * math.pi
        )

    pts.sort(key=__sort)
    return pts


def __ptl_distance(line: list[list], point: list, dx):
    """Compute the distance from a point to a line.

    :param line: Line defined by two points.

    :param point: Point.

    :param dx: Distance between the points that define the line.

    :return: Distance from the point to the line.
    """
    return (
        abs(
            (line[1][0] - line[0][0]) * (line[0][1] - point[1])
            - (line[1][1] - line[0][1]) * (line[0][0] - point[0])
        )
        / dx
    )


def __intersection(line1: list[list], line2: list[list]):
    """Return the intersection of `line1` and `line2`.

    If they don't intersect, the function returns (-1, -1).
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def __polyscore(cnt, pts, cen, alfa, beta):
    """Calculate the polyscore value."""
    # Too small area
    frame_area = cv2.contourArea(cnt)
    if frame_area < (4 * alfa * alfa) * 5:
        return 0

    gamma = alfa / 1.5

    pts = np.array(pts)

    # Too few points
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(cnt, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    pcnt = matplotlib.path.Path(pco.Execute(gamma)[0])
    wtfs = pcnt.contains_points(pts)
    pts_in_frame = min(np.count_nonzero(wtfs), 49)
    if pts_in_frame < min(pts.shape[0], 49) - 2 * beta - 1:
        return 0

    pcnt_in = pts[wtfs]
    hull = ConvexHull(pcnt_in).vertices
    points = pcnt_in[hull]

    # We are looking for the focal point of the cluster
    length = points.shape[0]
    sum_xy = np.sum(points, axis=0)
    cen2 = (sum_xy[0] / length, sum_xy[1] / length)

    # Distance between the group centroid and the frame centroid
    cen_dist = math.sqrt((cen[0] - cen2[0]) ** 2 + (cen[1] - cen2[1]) ** 2)

    lns = [
        [cnt[0], cnt[1]],
        [cnt[1], cnt[2]],
        [cnt[2], cnt[3]],
        [cnt[3], cnt[0]],
    ]
    i = 0
    j = 0
    for l in lns:
        d = math.sqrt((l[0][0] - l[1][0]) ** 2 + (l[0][1] - l[1][1]) ** 2)
        for p in points:
            r = __ptl_distance(l, p, d)
            if r < gamma:
                i += r
                j += 1
    if j == 0:
        return 0

    average_dist = i / j

    if frame_area == 0 or pts_in_frame == 0:
        return 0

    w_points = 1 + (average_dist / pts_in_frame) ** (1 / 3)
    w_centroid = 1 + (cen_dist / pts_in_frame) ** (1 / 5)
    return (pts_in_frame**4) / ((frame_area**2) * w_points * w_centroid)


def __padcrop(img, four_points):
    """Apply a border to the inner four points of the chessboard.

    This function applies a border to the inner four points of the
    chessboard in order to obtain a frame that contains the full board.
    """
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(four_points, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

    padded = pco.Execute(60)[0]

    debug.DebugImage(img).points(four_points, color=(0, 0, 255)).points(
        padded, color=(0, 255, 0)
    ).lines(
        [
            [four_points[0], four_points[1]],
            [four_points[1], four_points[2]],
            [four_points[2], four_points[3]],
            [four_points[3], four_points[0]],
        ],
        color=(255, 255, 255),
    ).lines(
        [
            [padded[0], padded[1]],
            [padded[1], padded[2]],
            [padded[2], padded[3]],
            [padded[3], padded[0]],
        ],
        color=(255, 255, 255),
    ).save(
        "cps_final_pad"
    )

    return __order_points(padded)


def cps(
    img: np.ndarray, points: list[list], lines: list[list]
) -> list[list[int]]:
    """Search for the chessboard position in the given image.

    :param img: Image to search.

    :param points: Points obtained in laps.

    :param lines: Lines detected by slid.

    :return: The four inner points of the detected chessboard.
    """
    ptp_cache = {}

    def ptp_distance(a, b):
        """Calculate the point-to-point distance.

        This function calculates the point-to-point distance with a
        cache to avoid multiple calculations.
        """
        idx = hash("__dis" + str(a) + str(b))
        if idx in ptp_cache:
            return ptp_cache[idx]
        ptp_cache[idx] = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return ptp_cache[idx]

    points = __check_correctness(__normalize(points), img.shape)

    # Clustering
    __points = {}
    points = __sort_points(points)
    __max = 0
    __points_max = []
    alfa = math.sqrt(cv2.contourArea(np.array(points)) / 49)
    X = DBSCAN(eps=alfa * 4).fit(points)
    for i in range(len(points)):
        __points[i] = []
    for i in range(len(points)):
        if X.labels_[i] != -1:
            __points[X.labels_[i]].append(points[i])
    for i in range(len(points)):
        if len(__points[i]) > __max:
            __max = len(__points[i])
            __points_max = __points[i]

    if len(__points) > 0 and len(points) > 49 / 2:
        points = __points_max

    n = len(points)
    beta = n * (5 / 100)  # beta = n * (100 - (CPS efectiveness))
    alfa = math.sqrt(cv2.contourArea(np.array(points)) / 49)

    # We are looking for the focal point of the cluster
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))

    def __v(l):
        y_0, x_0 = l[0][0], l[0][1]
        y_1, x_1 = l[1][0], l[1][1]

        x_2 = 0
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        a = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)][::-1]

        x_2 = img.shape[0]
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        b = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)][::-1]

        poly1 = __sort_points([[0, 0], [0, img.shape[0]], a, b])
        s1 = __polyscore(np.array(poly1), points, centroid, alfa / 2, beta)
        poly2 = __sort_points(
            [a, b, [img.shape[1], 0], [img.shape[1], img.shape[0]]]
        )
        s2 = __polyscore(np.array(poly2), points, centroid, alfa / 2, beta)

        return [a, b], s1, s2

    def __h(l):
        x_0, y_0 = l[0][0], l[0][1]
        x_1, y_1 = l[1][0], l[1][1]

        x_2 = 0
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        a = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)]

        x_2 = img.shape[1]
        t = (x_0 - x_2) / (x_0 - x_1 + 0.0001)
        b = [int((1 - t) * x_0 + t * x_1), int((1 - t) * y_0 + t * y_1)]

        poly1 = __sort_points([[0, 0], [img.shape[1], 0], a, b])
        s1 = __polyscore(np.array(poly1), points, centroid, alfa / 2, beta)
        poly2 = __sort_points(
            [a, b, [0, img.shape[0]], [img.shape[1], img.shape[0]]]
        )
        s2 = __polyscore(np.array(poly2), points, centroid, alfa / 2, beta)

        return [a, b], s1, s2

    pregroup = [[], []]  # Division into 2 groups (for the frame)
    for l in lines:  # We will review all of the lines
        # We reject lines that pass through the center of the cluster
        if __ptl_distance(l, centroid, ptp_distance(*l)) > alfa * 2.5:
            for p in points:
                # We check that the line passes near a good point
                if __ptl_distance(l, p, ptp_distance(*l)) < alfa:
                    # The line belongs to the ring
                    tx, ty = l[0][0] - l[1][0], l[0][1] - l[1][1]
                    if abs(tx) < abs(ty):
                        ll, s1, s2 = __v(l)
                        orientation = 0
                    else:
                        ll, s1, s2 = __h(l)
                        orientation = 1
                    if s1 == 0 and s2 == 0:
                        continue
                    pregroup[orientation].append(ll)

    pregroup[0] = __remove_duplicates(pregroup[0])
    pregroup[1] = __remove_duplicates(pregroup[1])

    if debug.DEBUG:
        # We create an outer ring
        def convex_approx(points, alfa=0.01):
            points = np.array(points)
            hull = ConvexHull(points).vertices
            cnt = points[hull]
            approx = cv2.approxPolyDP(
                cnt, alfa * cv2.arcLength(cnt, True), True
            )
            return __normalize(itertools.chain(*approx))

        ring = convex_approx(__sort_points(points))

        debug.DebugImage(img).lines(lines, color=(0, 0, 255)).points(
            points, color=(0, 0, 255)
        ).points(ring, color=(0, 255, 0)).points(
            [centroid], color=(255, 0, 0)
        ).save(
            "cps_debug"
        )

        debug.DebugImage(img).lines(pregroup[0], color=(0, 0, 255)).lines(
            pregroup[1], color=(255, 0, 0)
        ).save("cps_pregroups")

    score = {}  # Frame ranking with the result
    for v in itertools.combinations(pregroup[0], 2):  # Horizontal
        for h in itertools.combinations(pregroup[1], 2):  # Vertical
            poly = [
                __intersection(v[0], v[1]),
                __intersection(v[0], h[0]),
                __intersection(v[0], h[1]),
                __intersection(v[1], h[0]),
                __intersection(v[1], h[1]),
                __intersection(h[0], h[1]),
            ]
            poly = __check_correctness(poly, img.shape)
            if len(poly) != 4:
                continue
            poly = np.array(__sort_points(__normalize(poly)))
            if not cv2.isContourConvex(poly):
                continue
            score[-__polyscore(poly, points, centroid, alfa / 2, beta)] = poly

    score = collections.OrderedDict(sorted(score.items()))
    K = next(iter(score))

    inner_points = __normalize(score[K])
    inner_points = __order_points(inner_points)

    debug.DebugImage(img).points(points, color=(0, 255, 0)).points(
        inner_points, color=(0, 0, 255)
    ).points([centroid], color=(255, 0, 0)).lines(
        [
            [inner_points[0], inner_points[1]],
            [inner_points[1], inner_points[2]],
            [inner_points[2], inner_points[3]],
            [inner_points[3], inner_points[0]],
        ],
        color=(255, 255, 255),
    ).save(
        "cps_debug_2"
    )

    return __padcrop(img, inner_points)
