"""This is the straight-line-detector module."""


import math

import cv2
import numpy as np

from lc2fen.detectboard import debug


def __slid_segments(img):
    """Find all segments in the image using different settings.

    :param img: Image to search.

    :return: A list of all the segments found.
    """

    def detect_edges(img):
        """Apply Canny edge detector (automatic threshold)."""
        sigma = 0.25
        v = np.median(img)
        img = cv2.medianBlur(img, 5)
        img = cv2.GaussianBlur(img, (7, 7), 2)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(img, lower, upper)

    def simplify_image(img, limit, grid, iters):
        """Simplify image using CLAHE algorithm.

        This function simplifies an image using the CLAHE algorithm
        (adaptive histogram equalization).
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for _ in range(iters):
            img = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid).apply(
                img
            )
        debug.DebugImage(img).save("slid_clahe_@1")
        if limit != 0:
            kernel = np.ones((10, 10), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            debug.DebugImage(img).save("slid_clahe_@2")
        return img

    def detect_lines(img):
        """Detect lines using the probabilistic Hough transform."""
        beta = 2
        lines = cv2.HoughLinesP(
            img,
            rho=1,
            theta=np.pi / 360 * beta,
            threshold=40,
            minLineLength=50,
            maxLineGap=15,
        )  # [40, 40, 10]
        if lines is None:
            return []

        __lines = []
        for line in np.reshape(lines, (-1, 4)):
            __lines.append(
                [[int(line[0]), int(line[1])], [int(line[2]), int(line[3])]]
            )
        return __lines

    clahe_settings = [
        [3, (2, 6), 5],  # @1
        [3, (6, 2), 5],  # @2
        [5, (3, 3), 5],  # @3
        [0, (0, 0), 0],
    ]  # EE

    segments = []
    i = 0
    for key, arr in enumerate(clahe_settings):
        tmp = simplify_image(img, limit=arr[0], grid=arr[1], iters=arr[2])
        __segments = detect_lines(detect_edges(tmp))
        segments += __segments
        i += 1
        debug.DebugImage(detect_edges(tmp)).lines(__segments).save(
            "pslid_F%d" % i
        )
    return segments


def __scale_lines(raw_lines) -> list:
    """Scale raw_lines by a factor.

    :param raw_lines: Iterable of pairs of points.

        Note that a line is given by two points ((x1, y1), (x2, y2)).

    :return: List of the scaled lines.

        Each line is a pair of points.
    """
    lines = []
    s = 4

    def scale(x, y, s):
        return int(x * (1 + s) / 2 + y * (1 - s) / 2)

    for (x1, y1), (x2, y2) in raw_lines:
        x1 = scale(x1, x2, s)
        y1 = scale(y1, y2, s)
        x2 = scale(x2, x1, s)
        y2 = scale(y2, y1, s)
        lines.append(((x1, y1), (x2, y2)))

    return lines


def slid(img):
    """Detect the straight lines in the given image from the segments.

    :param img: Image to search.

    :return: List of the detected lines.

        Each line is a pair of points.
    """
    group = {}
    hashmap = {}
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

    def ptl_distance(line, point, dx):
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

    def similar_lines(line1, line2):
        """Determine whether `line1` is similar to `line2`."""
        da = ptp_distance(line1[0], line1[1])
        db = ptp_distance(line2[0], line2[1])

        d1a = ptl_distance(line1, line2[0], da)
        d2a = ptl_distance(line1, line2[1], da)
        d1b = ptl_distance(line2, line1[0], db)
        d2b = ptl_distance(line2, line1[1], db)

        # Average deviation from the straight line
        avg_dev = 0.25 * (d1a + d1b + d2a + d2b) + 0.00001

        # Allowed matching error
        delta = 0.0625 * (da + db)

        return da / avg_dev > delta and db / avg_dev > delta

    X = {}

    def __fi(x):
        if x not in X:
            X[x] = 0
        if X[x] == x or X[x] == 0:
            X[x] = x
        else:
            X[x] = __fi(X[x])
        return X[x]

    def __un(a, b):
        """Union & find."""
        ia, ib = __fi(a), __fi(b)
        X[ia] = ib
        group[ib] |= group[ia]

    def generate_points(a, b, n):
        """Return n equispaced points in segment given by a and b."""
        points = []
        t = 1 / n
        for i in range(n):
            x = a[0] + (b[0] - a[0]) * (i * t)
            y = a[1] + (b[1] - a[1]) * (i * t)
            points.append((int(x), int(y)))
        return points

    def merge_group(group, all_points):
        """Merge the group into a single line."""
        points = []
        for idx in group:
            points += generate_points(*hashmap[idx], n=10)

        all_points += points
        na_points = np.array(points)

        _, radius = cv2.minEnclosingCircle(na_points)
        w = radius * (math.pi / 2)
        vx, vy, cx, cy = cv2.fitLine(na_points, cv2.DIST_L2, 0, 0.01, 0.01)

        return (
            (int(cx - vx * w), int(cy - vy * w)),
            (int(cx + vx * w), int(cy + vy * w)),
        )

    # Find all segments in image
    segments = __slid_segments(img)

    # Divide segments into vertical and horizontal
    vh_segments = [[], []]
    for l in segments:
        h = hash(str(l))
        hashmap[h] = l
        group[h] = {h}
        X[h] = h

        t1 = l[0][0] - l[1][0]
        t2 = l[0][1] - l[1][1]
        if abs(t1) < abs(t2):  # If l is a vertical segment
            vh_segments[0].append(l)
        else:
            vh_segments[1].append(l)

    debug.DebugImage(img.shape).lines(
        vh_segments[0], color=debug.rand_color()
    ).lines(vh_segments[1], color=debug.rand_color()).save("slid_pre_groups")

    for lines in vh_segments:
        for i in range(len(lines)):
            l1 = lines[i]
            h1 = hash(str(l1))
            if X[h1] != h1:  # Line already grouped
                continue
            for j in range(i + 1, len(lines)):
                l2 = lines[j]
                h2 = hash(str(l2))
                if X[h2] != h2:
                    continue

                if similar_lines(l1, l2):
                    __un(h1, h2)

    if debug.DEBUG:
        __d = debug.DebugImage(img.shape)
        for i in group:
            if X[i] != i:
                continue
            ls = [hashmap[h] for h in group[i]]
            __d.lines(ls, color=debug.rand_color())
        __d.save("slid_all_groups")

    all_points = []
    raw_lines = []
    for i in group:
        if X[i] != i:
            continue
        raw_lines.append(merge_group(group[i], all_points))

    lines = __scale_lines(raw_lines)

    debug.DebugImage(img.shape).points(
        all_points, color=(0, 255, 0), size=2
    ).lines(raw_lines).save("slid_raw_lines")

    debug.DebugImage(img).lines(lines).save("slid_final")

    return lines
