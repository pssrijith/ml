import pandas as pd
import numpy as np
import warnings
from pandas import DataFrame, Series
from collections import defaultdict


def findClosestClustroid(clustroids, point):
    clustroid_mismatches = []
    for i, clustroid in enumerate(clustroids):
        clustroid_mismatches.append(calcPointMismatches(clustroid, point))
    closestClustroid = np.argmin(clustroid_mismatches)


def calcPointMismatches(pointA, pointB):
    if len(pointA) != len(pointB):
        raise Exception("Invalid points. Points should have same dimensions")
    mismatch = 0
    for i in range(len(pointA)):
        mismatch = mismatch + attrMismatch(pointA[i], pointB[i])
    return mismatch


def attrMismatch(a, b):
    if pd.isnull(a) and pd.isnull(b):
        return 0
    elif pd.isnull(a) or pd.isnull(b):
        return 1
    elif a == b:
        return 0
    else:
        return 1


class KModesClustering:

    def __init__(self, dataframe):
        if not isinstance(dataframe, DataFrame):
            raise Exception(
                "dataframe should be an instance of pandas.DataFrame")
        self.dataframe = dataframe
        (self.num_rows, self.num_attrs) = dataframe.shape

    ###
    ### the main cluster method. takes inputs 'k' - no.ofclusters and 'numIterations'
    ###
    def cluster(self, k, num_iter):
        print "k=%d,num_iter=%d".format(k, num_iter)

        cluster_attr_freq = [[defaultdict(int) for _ in range(self.num_attrs)]
                             for _ in range(k)]
        clustroids = self.initClustroids(k)
        cluster_assignment = self.assignClustroids(k)
        i = 0
        while i < num_iter:
            prev_clustroids = clustroids
            prev_cluster_assignment = cluster_assignment

            clustroids = self.reComputeClustroids(k)
            if prev_clustroids == clustroids:
                print "Reached clustroid equilibrium"
                break
            cluster_assignment = self.assignClustroids(k, clustroids)
            i = i + 1
        print "Done clustering i = ", i

    ###
    ### initClustroids(). Randomly initializes k points as centroids trying to keep the poins as
    ### far from each other
    ###
    def initClustroids(self, k):
        clustroids = []
        clustroids.append(self.getRandomRow())
        while (len(clustroids) < k):
            mismatches = self.computeMismatchesWithClustroids(clustroids)
            norm_mismatches = mismatches / mismatches.sum()
            norm_mismatches.sort()
            # get the next random clustroid by finding the next largest of the mismatches
            # above a random value from the current clustroid points
            next_clust_Idx = norm_mismatches[norm_mismatches.cumsum() >= np.random.rand()].index[0]
            clustroids.append(self.dataframe[self.dataframe.columns].iloc[next_clust_Idx, :])
        clustroids = DataFrame(clustroids, columns=self.dataframe.columns)

    ###
    ### assignClustroids : to assign each input to its nearest cluster. in our case, the cluster
    ### that has the lowest mismatch
    ###
    def assignClustroids(self, clustroids, cluster_attr_freq, prev_cluster_assignment=None):
        cluster_assignment = []
        for rowIdx, row in self.dataframe.iterrows:
            closest_clustroid = findClosestClustroid(clustroids, point=row)
            cluster_assignment.append(closest_clustroid)
            if prev_cluster_assignment != None and \
                            prev_cluster_assignment[rowIdx] == cluster_assignment[rowIdx]:
                continue # no need to update cluster_attr_freq if the point did not move to new cluster
            # update cluster_attr_freq for current cluster
            for attr_name, attr_val in row.iteritems() :
                # increment the frequency count of the attr vals in the cluster
                cluster_attr_freq[closest_clustroid][attr_name][attr_val] += 1
                 # decrement frequencies of the old cluster where this point was assigned
                if prev_cluster_assignment != None :
                    cluster_attr_freq[prev_cluster_assignment[rowIdx]][attr_name][attr_val] -= 1
        return cluster_assignment


    ###
    ### recomputeClustroids - recalculate each cluster's mode (most frequent attributes of each column) . The mode will be the new clustroid
    ###
    def reComputeClustroids(self, k):
        newClustroids = []

        for i in range(k):
            # get the indexes of clustroid 'i' from the clustroid_assignment df. This will return
            # indexes of all points assigned to clustroid `i`
            idx = self.clustroid_assignment[self.clustroid_assignment['clustroids'] == i].index
            print "DEBUGGG:: idx.ize = ", idx.size

            newMode = self.getMode(self.dataframe[self.dataframe.index.isin(idx)])
            newClustroids.append(newMode)

        # replace current clustroids with newClustroids
        self.clustroids = DataFrame(newClustroids, columns=self.dataframe.columns);

    ###
    ### getMode : returns mode of an input dataframe.
    ###
    def getMode(self, df):
        df_mode = df.mode()
        ## the mode call returns a dataframe with possibly more than one mode. 
        ## multiple modes happen when 2 modes are equally frequent. we pick the first mode using iloc
        return df_mode[df.columns].iloc[0, :]

    def getRandomRow(self):
        idx = np.random.random_integers(0, self.num_rows)
        return self.dataframe[self.dataframe.columns].iloc[idx, :]

    def computeMismatchesWithClustroids(self, clustroids):
        result = None
        for clustroid in clustroids:
            if result is None:
                result = self.calcDFMismatchWithPoint(clustroid)
            else:
                result = pd.concat(
                    [result, self.calcDFMismatchWithPoint(clustroid)], axis=1).min(axis=1)
        return result

    def calcDFMismatchWithPoint(self, point):
        # return mismatch of data frame with the given point. compares the point with each row in the dataframe and
        # returns a ndarray of mismatch counts in index corresponding to each row comparison
        df_mismatch = pd.Series(index=np.arange(self.num_rows))
        for i in range(self.num_rows):
            df_mismatch[i] = self.calcPointMismatches(
                self.dataframe[self.dataframe.columns].iloc[i, :], point)
        return df_mismatch
