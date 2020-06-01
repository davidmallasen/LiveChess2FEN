"""
Work with an image in the iterative process of finding a chessboard.
"""
import math

import cv2
import numpy as np


def image_scale(pts, scale):
    """Scale to original image size."""
    return [[x / scale, y / scale] for (x, y) in pts]


def image_resize(img, height=500):
    """Resize image to same normalized area (height**2)."""
    shape = np.shape(img)
    scale = math.sqrt((height * height) / (shape[0] * shape[1]))
    img = cv2.resize(img, (int(shape[1] * scale), int(shape[0] * scale)))
    return img, shape, scale


def image_transform(img, points):
    """Crop original image using perspective warp."""
    board_length = 1200

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [board_length, 0], [board_length, board_length],
                       [0, board_length]])
    mat = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, mat, (board_length, board_length))


class ImageObject:
    """
    Represents an image object in the iterative process of finding the
    chessboard.
    """

    def __init__(self, img):
        """Save and prepare image array."""
        # Downscale for speed
        downscaled_img_, shape_, scale_ = image_resize(img)

        # We save the whole sequence of transformations
        # attribute[i] is the attribute of iteration i, being iteration
        # 0 the first one
        self.images = [{'orig': img, 'main': downscaled_img_}]
        self.shape = [shape_]  # (0, 0)
        self.scale = [scale_]  # 1
        self.points = []  # Points of the new cropped image for next iteration

    def __getitem__(self, attr):
        """Return last image as array."""
        return self.images[-1][attr]

    def __setitem__(self, attr, val):
        """Save image to object as last image."""
        self.images[-1][attr] = val

    def add_image(self, img):
        """Add a new image in the iteration."""
        # Downscale for speed
        downscaled_img_, shape_, scale_ = image_resize(img)

        self.images.append({'orig': img, 'main': downscaled_img_})
        self.shape.append(shape_)
        self.scale.append(scale_)

    def crop(self, pts):
        """Crop using 4 points transform."""
        pts_orig = image_scale(pts, self.scale[-1])
        img_crop = image_transform(self.images[-1]['orig'], pts_orig)
        self.points.append(pts_orig)
        self.add_image(img_crop)

    def get_images(self):
        """Return images list."""
        return self.images

    def get_points(self):
        """Return points list."""
        return self.points
