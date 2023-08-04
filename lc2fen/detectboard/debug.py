"""This is the module for debugging utilities."""


import itertools
from copy import copy
from random import randint

import cv2
import numpy as np


DEBUG = False  # Set it to `True`/`False` to enable/disable debug images
COUNTER = itertools.count()
DEBUG_SAVE_DIR = "data/boards/debug_steps/"


def rand_color():
    """Return a random rgb color."""
    return randint(0, 255), randint(0, 255), randint(0, 255)


class DebugImage:
    """Represent a debug image.

    This class is used for drawing points and lines and saving the
    resulting image.
    """

    def __init__(self, img):
        """Initialize an instance of the `DebugImage`."""
        if DEBUG:
            if isinstance(img, tuple):
                img = np.zeros((img[0], img[1], 3), np.uint8)
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.img = copy(img)

    def lines(self, _lines, color=(0, 0, 255), size=2):
        """Draw lines in the image."""
        if DEBUG:
            for li1, li2 in _lines:
                cv2.line(self.img, tuple(li1), tuple(li2), color, size)
        return self

    def points(self, _points, color=(0, 0, 255), size=10):
        """Draw points in the image."""
        if DEBUG:
            for point in _points:
                cv2.circle(
                    self.img, (int(point[0]), int(point[1])), size, color, -1
                )
        return self

    def save(self, filename, prefix=True):
        """Save the image."""
        global COUNTER
        if DEBUG:
            if prefix:
                __prefix = "__debug_" + "%04d" % int(next(COUNTER)) + "_"
            else:
                __prefix = ""

            cv2.imwrite(
                DEBUG_SAVE_DIR + __prefix + filename + ".jpg", self.img
            )
