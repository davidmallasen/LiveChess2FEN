"""This is the lattice-points-search module."""


import collections

import cv2
import numpy as np
import onnxruntime
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist

from lc2fen.detectboard import debug, image_object
from lc2fen.detectboard import poly_point_isect


__LAPS_SESS = onnxruntime.InferenceSession(
    "lc2fen/detectboard/models/laps_model.onnx"
)

__ANALYSIS_RADIUS = 10


def __find_intersections(lines):
    """Find all intersections."""
    __lines = [[(a[0], a[1]), (b[0], b[1])] for a, b in lines]
    return poly_point_isect.isect_segments(__lines)


def __cluster_points(points, max_dist=10):
    """Cluster very similar points."""
    link_matrix = single(pdist(points))
    cluster_ids = fcluster(link_matrix, max_dist, "distance")

    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(cluster_ids):
        clusters[cluster_id].append(points[i])
    clusters = clusters.values()
    # If two points are close, they become one mean point
    clusters = map(
        lambda arr: (
            np.mean(np.array(arr)[:, 0]),
            np.mean(np.array(arr)[:, 1]),
        ),
        clusters,
    )
    return list(clusters)


def __is_lattice_point(img):
    """Determine if a point is a lattice point."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    img = cv2.Canny(img, 0, 255)
    img = cv2.resize(img, (21, 21), interpolation=cv2.INTER_CUBIC)

    # Geometric detector to filter easy points
    img_geo = cv2.dilate(img, None)
    mask = cv2.copyMakeBorder(
        img_geo,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    _c = np.zeros((23, 23, 3), np.uint8)
    num_rhomboid = 0
    for cnt in contours:
        _, radius = cv2.minEnclosingCircle(cnt)
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and radius < 14:
            cv2.drawContours(_c, [cnt], 0, (0, 255, 0), 1)
            num_rhomboid += 1
        else:
            cv2.drawContours(_c, [cnt], 0, (0, 0, 255), 1)

    if num_rhomboid == 4:
        return True

    # Neural detector if unable to decide using the geometric detector
    X = [np.where(img > int(255 / 2), 1, 0).ravel()]
    X = X[0].reshape([-1, 21, 21, 1]).astype("float32")

    pred = __LAPS_SESS.run(None, {__LAPS_SESS.get_inputs()[0].name: X})[0][0]

    return pred[0] > pred[1] and pred[1] < 0.03 and pred[0] > 0.975


def laps(img: np.ndarray, lines):
    """Search for the lattice points in the given image.

    :param img: Image to search.

    :param lines: Lines detected by slid.

    :return: Points detected to be part of the chessboard grid.
    """
    intersection_points = __find_intersections(lines)

    debug.DebugImage(img).lines(lines, color=(0, 0, 255)).points(
        intersection_points, color=(255, 0, 0), size=2
    ).save("laps_in_queue")

    points = []
    for pt in intersection_points:
        # Pixels are in integers
        pt = (int(pt[0]), int(pt[1]))

        if pt[0] < 0 or pt[1] < 0:
            continue

        # Size of our analysis area
        lx1 = max(0, int(pt[0] - __ANALYSIS_RADIUS - 1))
        lx2 = max(0, int(pt[0] + __ANALYSIS_RADIUS))
        ly1 = max(0, int(pt[1] - __ANALYSIS_RADIUS))
        ly2 = max(0, int(pt[1] + __ANALYSIS_RADIUS + 1))

        # Cropping for detector
        dimg = img[ly1:ly2, lx1:lx2]
        dimg_shape = np.shape(dimg)

        # Not valid
        if dimg_shape[0] <= 0 or dimg_shape[1] <= 0:
            continue

        # Detect if it is a lattice point
        if not __is_lattice_point(dimg):
            continue

        points.append(pt)

    if points:
        points = __cluster_points(points)

    debug.DebugImage(img).points(
        intersection_points, color=(0, 0, 255), size=3
    ).points(points, color=(0, 255, 0)).save("laps_good_points")

    return points


def check_board_position(
    img: np.ndarray, board_corners: list[list[int]], tolerance: int = 20
):
    """Check if chessboard is in position given by the board corners.

    :param img: Image to check.

    :param board_corners: List of the coordinates of four board corners.

    :param tolerance: Number of lattice points that must be correct.

    :return: A pair formed by a boolean indicating if the chessboard is
    in the position given by the board corners and the cropped image.
    """
    # We will check the interior 6x6 square grid lattice points of the
    # cropped 500x500 image, as done by LAPS
    cropped_img = image_object.image_transform(img, board_corners)

    correct_points = 0
    for row_corner in range(150, 1200, 150):
        for col_corner in range(150, 1200, 150):
            # Size of our analysis area
            lx1 = max(0, int(row_corner - __ANALYSIS_RADIUS - 1))
            lx2 = max(0, int(row_corner + __ANALYSIS_RADIUS))
            ly1 = max(0, int(col_corner - __ANALYSIS_RADIUS))
            ly2 = max(0, int(col_corner + __ANALYSIS_RADIUS + 1))

            # Cropping for detector
            dimg = cropped_img[ly1:ly2, lx1:lx2]
            dimg_shape = np.shape(dimg)

            # Not valid
            if dimg_shape[0] <= 0 or dimg_shape[1] <= 0:
                continue

            # Detect if it is a lattice point
            if __is_lattice_point(dimg):
                correct_points += 1

    return correct_points >= tolerance, cropped_img
