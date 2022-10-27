# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""

import numpy as np
from core.point import Point
from core.distance import Distance
from core.cluster import Cluster
import random as rand

class ClusterSet():

    def __init__(self):
        self.__clusters = []  # List of Clusters

    @property
    def clusters(self):
        return self.__clusters

    def __len__(self):
        return len(self.__clusters)

    def add_point_at_cluster_number(self, point, cluster_id):
        self.__clusters[cluster_id].add_point(point)

    def populate(self, points, k):
        if (k > len(points)):
            raise Exception('The number of clusters cannot be greater than the \
                number of records in the dataset.')
        population = range(0, len(points))
        # pick with no replacements !
        random_indexes = rand.sample(population, k)

        for i in random_indexes:
            c = Cluster()
            c.set_center_point(points[i])
            self.clusters.append(c)

if __name__ == '__main__':
    pass
