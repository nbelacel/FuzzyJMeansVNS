# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""


from scipy.spatial import distance
import numpy as np


class Point:
    dimension = 3  # Static/class var
    nextPid = 0

    def __init__(self):
        self.__groupNo = None
        self.__featureSet = None
        self.__label = None
        self.__dimension = None
        self.__pid = Point.nextPid + 1
        self.__inc_nextPid()
        self.__avgClusterDist = 0.0

    @property
    def avgClusterDist(self):
        return self.__avgClusterDist

    @avgClusterDist.setter
    def avgClusterDist(self, val):
        self.__avgClusterDist = val

    @property
    def pid(self):
        return self.__pid

    @property
    def groupNo(self):
        return self.__groupNo

    @groupNo.setter
    def groupNo(self, val):
        self.__groupNo = val

    @property
    def featureSet(self):
        return self.__featureSet

    @featureSet.setter
    def featureSet(self, val):
        self.__featureSet = val

    @property
    def label(self):
        return self.__label

    def setFeature(self, i, v):
        if i >= Point.dimension:
            raise Exception('This feature index is out of range.')
        else:
            self.featureSet[i] = v

    def __inc_nextPid(self):
        Point.nextPid + 1
        Point.nextPid = Point.nextPid + 1

    def parse_point(self, vector):
        self.__featureSet = vector[0: Point.dimension]
        self.__label = vector[-1]

    def get_distance(self, other_point):
        return self.__get_euclidian_distance(self.featureSet, other_point.featureSet)

    def __get_euclidian_distance(self, vec_1, vec_2):
        return distance.euclidean(vec_1, vec_2)

    def get_is_equal(self, other_point):
        return np.array_equal(self.featureSet, other_point.featureSet)

    def get_feature(self, index):
        if index >= Point.dimension:
            raise ValueError('This feature index is out of range')
        else:
            return self.featureSet[index]


if __name__ == '__main__':
    pass