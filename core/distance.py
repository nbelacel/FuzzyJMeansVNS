# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""


from scipy.spatial import distance as ed
from core.point import Point
import numpy
import math


class Distance():
    preferredMetric = 'EUCLIDEAN'  # can also be KERNEL

    def __validate_2d_matrix(self, data):
        # Make sure that data is a numpy array
        if type(data) != numpy.ndarray:
            raise Exception('Expecting data to be of type numpy.ndarray !')
        if (len(data.shape) != 2):
            raise Exception('data is not in the shape of a 2D matix!')

    def __kmeans_plus_distance(self, a, b):
        if type(a) != numpy.ndarray:
            raise Exception('Expecting a to be of type numpy.ndarray !. Type was {}'.format(type(a)))
        if (len(a.shape) != 1):
            raise Exception('a is not in the shape of a 1D matix!')

        if type(b) != numpy.ndarray:
            raise Exception('Expecting b to be of type numpy.ndarray !. Type was {}'.format(type(b)))
        if (len(b.shape) != 1):
            raise Exception('b is not in the shape of a 1D matix!')

        distance = 0.0
        for i in range(0, Point.dimension):
            distance += (a[i] - b[i]) * (a[i] - b[i])
        distance = math.sqrt(distance)

        return distance

    def get_avg(self, data):
        self.__validate_2d_matrix(data)
        # Select all rows and all columns up to Point.dimension
        # by specifying ‘:’ for in the rows index, and :Point.dimension
        # in the columns index.
        X = data[:, :Point.dimension]
        # Example return value : [2.5 3.5 4.5]
        return X.sum(axis=0)/X.shape[0]

    def get_sigma(self, data):
        self.__validate_2d_matrix(data)
        X = data[:, :Point.dimension]
        avg = self.get_avg(data)
        data_size = X.shape[0]  # Get nubmer of columns
        sigma = 0.0
        for i in range(0, data_size):
            sigma += self.__kmeans_plus_distance(data[i], avg) * self.__kmeans_plus_distance(data[i], avg)
        sigma = math.sqrt(sigma / data_size)
        return sigma

    def get_distance(self, a, b):
        if Distance.preferredMetric == 'EUCLIDEAN':
            return self.__kmeans_plus_distance(a, b)
        elif Distance.preferredMetric == 'KERNEL':
            return self.kernelDistance(a, b)
        else:
            return 0

    def kernelDistance(self, a, b):
        raise Exception("to implement")

    # def scipy_euclidean_distance(self, a, b):
    #     return ed.euclidean(a, b)


if __name__ == '__main__':
    pass