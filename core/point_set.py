# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""

import numpy as np
from core.point import Point


class PointSet():

    def __init__(self):
        self.__points = []

    @property
    def points(self):
        return self.__points

    def __len__(self):
        return len(self.__points)

    def populate(self, vectors):
        data = vectors.copy()
        for i in range(0, len(data)):
            p = Point()
            p.parse_point(data[i])
            self.__points.append(p)


if __name__ == '__main__':
    pass