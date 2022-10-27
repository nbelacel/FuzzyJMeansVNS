# -*- coding: utf-8 -*-
"""
National Research Council Canada
COPYRIGHT:    2019 National Research Council Canada
"""


from core.distance import Distance
from core.cluster import Cluster
from core.point import Point
import numpy as np
import sys
import math


def avgPoint(pointSet):
    avg = Point()
    avg.parse_point(np.zeros(Point.dimension))

    for i in range(0, len(pointSet)):
        for j in range(0, Point.dimension):
            val = avg.featureSet[j] + pointSet.points[i].featureSet[j]
            avg.setFeature(j, val)

        for j in range(0, Point.dimension):
            avg.setFeature(j, avg.featureSet[j]/len(pointSet))

    return avg


def patternVariance(pointSet):
    avg = avgPoint(pointSet)
    variance = Point()
    variance.parse_point(np.zeros(Point.dimension))

    for i in range(0, len(pointSet)):
        for j in range(0, Point.dimension):
            val = variance.featureSet[j] + math.pow(pointSet.points[i].featureSet[j] - avg.featureSet[j], 2.0)
            variance.setFeature(j, val)

    for i in range(0, Point.dimension):
        avg.setFeature(j, avg.featureSet[j]/len(pointSet))

    return variance


def scat(clusterSet, pointSet, matrix, fuzzyConstant):
    sigmaX = patternVariance(pointSet)
    normSigmaX = 0

    for j in range(0, Point.dimension):
        normSigmaX += math.sqrt(math.pow(sigmaX.featureSet[j], 2.0))

    return leastSquaredError(clusterSet, pointSet, matrix, fuzzyConstant) / (len(clusterSet) * len(pointSet) * normSigmaX)


def dist(clusterSet, pointSet):
    dMax = maxInterClusterDistance(clusterSet)
    dMin = minInterClusterDistance(clusterSet)
    s1 = 0.0

    for k in range(0, len(clusterSet)):
        s2 = 0.0
        for z in range(0, len(clusterSet)):
            d = Distance().get_distance(clusterSet.clusters[k].featureSet, clusterSet.clusters[z].featureSet)
            s2 += d
        s1 += 1/s2

    return (dMax/dMin) * s1


def composeWithinBetweenIndex(clusterSet, pointSet, matrix, fuzzyConstant):
    alpha = 1.0	# = dist(cMax);

    return (alpha*scat(clusterSet, pointSet, matrix, fuzzyConstant)) + dist(clusterSet, pointSet)


def PBMIndex(clusterSet, pointSet, matrix, fuzzyConstant):
    index = 0.0
    E1 = 0.0
    Ek = 0.0
    Dk = 0.0

    avg = avgClusterCenter(clusterSet)

    for i in range(0, len(clusterSet)):
        for j in range(len(clusterSet)):
            d = Distance().get_distance(clusterSet.clusters[i].featureSet, clusterSet.clusters[j].featureSet)
            if d > Dk:
                Dk = d

        for k in range(0, len(clusterSet)):
            for i in range(0, len(pointSet)):
                d = Distance().get_distance(clusterSet.clusters[k].featureSet, pointSet.points[i].featureSet)
                Ek += matrix.data[i][k] * d

        for i in range(0, len(pointSet)):
            d = Distance().get_distance(avg.featureSet, pointSet.points[i].featureSet)
            E1 += d

    index = ((1.0 / len(clusterSet)) * (E1 / Ek) * Dk)

    return index * index


def kwonIndex(clusterSet, pointSet, matrix, fuzzyConstant):
    avg = avgClusterCenter(clusterSet)
    punish = 0.0
    for k in range(0, len(clusterSet)):
        dist = Distance().get_distance(clusterSet.clusters[k].featureSet, avg.featureSet)
        punish += pow(dist, 2.0)
    return (leastSquaredError(clusterSet, pointSet, matrix, fuzzyConstant) + (punish / len(clusterSet))) / pow(minInterClusterDistance(clusterSet), 2.0)


def fukuyamaSugenoIndex(clusterSet, pointSet, matrix, fuzzyConstant):
    avg = avgClusterCenter(clusterSet)
    km = 0.0

    for i in range(0, len(pointSet)):
        for k in range(0, len(clusterSet)):
            num_1 = math.pow(matrix.data[i][k], float(fuzzyConstant))
            dist = Distance().get_distance(pointSet.points[i].featureSet, avg.featureSet)
            num_2 = math.pow(dist, 2.0)
            km = km + (num_1 * num_2)

    return leastSquaredError(clusterSet, pointSet, matrix, fuzzyConstant) - km


def avgClusterCenter(clusterSet):
    avg = Cluster()
    point = Point()
    point.parse_point(np.zeros(Point.dimension))
    avg.parse_point(point)

    for i in range(0, len(clusterSet)):
        for j in range(0, Point.dimension):
            val = avg.featureSet[j] + clusterSet.clusters[i].featureSet[j]
            avg.setFeature(j, val)

    for j in range(0, Point.dimension):
        avg.setFeature(j, avg.featureSet[j]/len(clusterSet))

    return avg


def leastSquaredError(clusterSet, pointSet, matrix, fuzzyConstant):
    dist_calc = Distance()
    jm = 0.0
    for i in range(0, len(pointSet)):
        for k in range(0, len(clusterSet)):
            first = math.pow(matrix.data[i][k], float(fuzzyConstant))
            dist = dist_calc.get_distance(pointSet.points[i].featureSet, clusterSet.clusters[k].featureSet)
            second = math.pow(dist, 2.0)

            jm += (first * second)
    return jm


def maxInterClusterDistance(clusterSet):
    maxDist = 0.0
    dist_calc = Distance()

    for k1 in range(0, len(clusterSet)):
        for k2 in range(0, len(clusterSet)):
            if k1 == k2:
                continue

            dist = dist_calc.get_distance(clusterSet.clusters[k1].featureSet, clusterSet.clusters[k2].featureSet)
            if (dist > maxDist):
                maxDist = dist

    return maxDist


def minInterClusterDistance(clusterSet):
    minDist = sys.float_info.max
    dist_calc = Distance()

    for k1 in range(0, len(clusterSet)):
        for k2 in range(0, len(clusterSet)):
            if k1 == k2:
                continue

            dist = dist_calc.get_distance(clusterSet.clusters[k1].featureSet, clusterSet.clusters[k2].featureSet)
            if (dist < minDist):
                minDist = dist

        return minDist


def xieBeniIndex(clusterSet, pointSet, matrix, fuzzyConstant):
    return leastSquaredError(clusterSet, pointSet, matrix, fuzzyConstant) / (len(pointSet) * math.pow(minInterClusterDistance(clusterSet), 2.0))


def avgDistanceWithCluster(p, c):
    dist = 0.0
    dist_calc = Distance()
    for i in range(0, len(c.points)):
        dist += dist_calc.get_distance(p.featureSet, c.points[i].featureSet)

    return dist/c.numPoints


def silhouette(clusterSet, pointSet):
    s = 0.0
    for k in range(0, len(clusterSet)):
        for i in range(0, clusterSet.clusters[k].numPoints):
            p = clusterSet.clusters[k].points[i]
            a = avgDistanceWithCluster(p, clusterSet.clusters[k])
            b = sys.float_info.max

            for m in range(0, len(clusterSet)):
                if m != k:
                    avgDist = avgDistanceWithCluster(p, clusterSet.clusters[m])
                    if avgDist < b:
                        b = avgDist
            if a < b:
                s += 1 - (a / b)
            elif b < a:
                s += (b / a) - 1

    return s / len(pointSet)


if __name__ == '__main__':
    pass
