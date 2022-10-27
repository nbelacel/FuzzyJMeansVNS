# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""

from core.point import Point
from core.distance import Distance
import math
import numpy as np


class Cluster():
    outlier = 10
    link = 0.5
    threshold = 0.01
    nextGroupNo = 0

    def __init__(self):
        self.__inc_cluster_groupNo()
        self.__points = []
        self.__featureSet = None
        self.__groupNo = None

        self.__siblingSet = []
        self.__population = 0     # number of data points assigned to this cluster
        self.__longest = 0.0      # distance of the farthest point from the cluster center
        self.__fpt = None         # the farthest point from the center in the cluster
        self.__sigma = 0          # standard deviation
        self.__avgDist = None     # average distance of points to centroid deviation

    def setFeature(self, i, v):
        if i >= Point.dimension:
            raise Exception('This feature index is out of range.')
        else:
            self.featureSet[i] = v

    @property
    def groupNo(self):
        return self.__groupNo

    @groupNo.setter
    def groupNo(self, val):
        self.__groupNo = val

    @property
    def population(self):
        return self.__population

    @property
    def featureSet(self):
        return self.__featureSet

    @property
    def points(self):
        return self.__points
    
    @property
    def numPoints(self):
        return self.__getNumPoints()

    @property
    def sigma(self):
        return self.__sigma

    @property
    def has_outlier(self):
        if self.__longest > self.__sigma * Cluster.outlier:
            return True
        else:
            return False

    @property
    def fpt(self):
        return self.__fpt

    def parse_point(self, point):
        self.__featureSet = point.featureSet[0: Point.dimension]

    def __check_is_point(self, val):
        if (type(val) != Point):
            raise Exception('Expecting val to be a Point')
        # Make sure that val is a numpy array
        if type(val.featureSet) != np.ndarray:
            raise Exception('Expecting point featureSet to be of type numpy.ndarray !')
        if (len(val.featureSet.shape) != 1):
            raise Exception('point featureSet is not a vector')

    def __inc_cluster_groupNo(self):
        Cluster.nextGroupNo += 1

    def __getNumPoints(self):
        return len(self.__points)

    def set_center_point(self, val):
        self.__check_is_point(val)
        self.__featureSet = val.featureSet.copy()

    def add_point(self, point):
        self.__check_is_point(point)

        # distance between current cluster centroid and the point we are adding
        distance = Distance().get_distance(point.featureSet, self.featureSet)

        cluster_features = self.featureSet
        # print('=======================================')
        # print('Point to Add : {}'.format(point.featureSet))
        # print('Current Cluster Val : {}'.format(self.featureSet))
        # print('Distance in add_point : {}'.format(distance))
        # print('Population in add_point : {}'.format(self.__population))

        for i in range(0, Point.dimension):
            cluster_features[i] = (self.__population * cluster_features[i] + point.featureSet[i]) / (self.__population + 1)

        if distance > self.__longest:
            self.__fpt = point
            self.__longest = distance

        if len(self.__points) == 0:
            point.avgClusterDist = 0
        else:
            avgDist = 0.0
            for i in range(0, self.__getNumPoints()):
                dist = Distance().get_distance(point.featureSet, self.__points[i].featureSet)
                avgDist += (dist - avgDist) / (i + 1)
                self.__points[i].avgClusterDist += (dist - self.__points[i].avgClusterDist) / (self.__population)

            point.avgClusterDist = avgDist

        self.__sigma = math.sqrt((self.__sigma * self.__sigma * self.__population + distance * distance) / (self.__population + 1))
        self.__population = self.__population + 1
        self.__points.append(point)
        # print('++++ Sigma in add_point : {}'.format(self.__sigma))
        # print('=======================================')

    def is_shifted(self):
        for i in range(0, Point.dimension):
            if self.featureSet[i] - self.featureSet[i] >= self.threshold:
                return True
        return False

    # remove the farthest point from the cluster center
    def remove_fpt(self):
        for i in range(0, Point.dimension):
            self.featureSet[i] = (self.__population * self.featureSet[i] - self.__fpt.featureSet[i]) / (self.__population - 1)

        self.__sigma = math.sqrt((self.__sigma * self.__sigma * self.__population - self.__longest * self.__longest) / (self.__population - 1))
        self.__population = self.__population - 1

        for i in range(0, len(self.__points)):
            p = self.__points[i]
            if (self.__fpt == p):
                self.__points.pop(i)

        self.__fpt = None

    # update the cluster and reset dynamic parameters
    def update(self):
        for i in range(0, Point.dimension):
            # What does this do in C++ ...?
            self.featureSet[i] = self.featureSet[i]
            self.featureSet[i] = 0

    # set link number for cluster
    def setLinkNo(self, no):
        if self.groupNo > no:
            self.groupNo = no
            for i in range(0, len(self.__siblingSet)):
                self.__siblingSet[i].setGroupNo(no)

    def can_merge(self, cluster):
        interDist = Distance().get_distance(self.featureSet, cluster.featureSet)
        sigma1 = cluster.sigma
        
        if interDist <= (self.__sigma + sigma1) * Cluster.link:
            return True
        else:
            return False

    def addSibling(self, cluster):
        self.__siblingSet.append(cluster)

    def link_with(self, cluster):
        groupNo1 = self.groupNo
        groupNo2 = cluster.groupNo

        if groupNo1 < groupNo2:
            cluster.setLinkNo(groupNo1)
        elif (groupNo2 < groupNo1):
            self.setLinkNo(groupNo2)

        self.__siblingSet.append(cluster)
        cluster.addSibling(self)


if __name__ == '__main__':
    pass
