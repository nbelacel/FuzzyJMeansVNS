# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""
import numpy as np
from core.point_set import PointSet
from core.cluster_set import ClusterSet


class Matrix():
    def __init__(self, clusterSet, pointSet):
        self.__clusterSet = clusterSet
        self.__pointSet = pointSet
        self.__rows = len(pointSet.points)
        self.__cols = len(clusterSet.clusters)
        self.__matrix = np.zeros(shape=(self.__rows, self.__cols))
        self.__populate_matrix()

    @property
    def data(self):
        return self.__matrix

    def __populate_matrix(self):
        for k in range(0, self.__cols):
            for i in range(0, self.__clusterSet.clusters[k].numPoints):
                self.__clusterSet.clusters[k].points[i].groupNo = k

        for i in range(0, self.__rows):
            self.__matrix[i][self.__pointSet.points[i].groupNo] = 1

    def print_matrix(self):
        print('\n------------------------------\n')
        print("non-transposed partition matrix:")
        print(self.__matrix)
        print()

    def print_transposed_matrix(self):
        print('\n------------------------------\n')
        print("non-transposed partition matrix:")
        print(self.__matrix.transpose())
        print()


if __name__ == '__main__':
    pass