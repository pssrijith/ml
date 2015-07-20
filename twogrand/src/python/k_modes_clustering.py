import pandas as pd
import numpy as np
from pandas import DataFrame
from collections import defaultdict


def calc_attr_mismatch(a, b):
    """
     compares two attributes and computes mismatch count. if both inputs are nullthen it would return "0 mismatch"
     unlike how pd.mode counts. this is specific to the input dataset for twogrand where we have many questions that
     users have skipped intentionally or not seen yet.

    :rtype : int - the number of mismatches (distance)
    """
    if pd.isnull(a) and pd.isnull(b):
        return 0
    elif pd.isnull(a) or pd.isnull(b):
        return 1
    elif a == b:
        return 0
    else:
        return 1


def calc_point_mismatches(point_a, point_b):
    """
    compares two points and computes their mismatch count.
    :param point_a:
    :param point_b:
    :rtype : int
    """
    if len(point_a) != len(point_b):
        raise Exception("Invalid points. Points should have same dimensions")
    mismatch = 0
    for i in range(len(point_a)):
        mismatch += calc_attr_mismatch(point_a[i], point_b[i])
    return mismatch


def find_closest_clustroid(clustroids, point):
    """
    returns the closes clustroid to the point from a list of clustroids along with the distance(mismatches) from the
    point to this clustroid
    :param clustroids:
    :param point:
    :rtype : closest_clustroid
             - the cluster index that is closest to the point
             distance
             - the distance of the point to the closest clustroid. useful to compute cluster cohesion
    """
    clustroid_mismatches = []
    for i, clustroid in clustroids.iterrows():
        clustroid_mismatches.append(calc_point_mismatches(clustroid, point))
    closest_clustroid = np.argmin(clustroid_mismatches)
    distance = clustroid_mismatches[closest_clustroid]
    return closest_clustroid, distance


def get_attr_mode(attr_dict):
    """
    This method returns the mode attribute (attribute that has the maximum frequency
    :param attrDict : is a dictionary that contains the attr as key and corresponding frequency as the values
    """
    attr_mode = list(attr_dict.keys())[np.argmax(list(attr_dict.values()))]
    return attr_mode


def frames_equal(df1, df2):
    if not isinstance(df1, DataFrame) or not isinstance(df2, DataFrame):
        raise Exception(
            "dataframe should be an instance of pandas.DataFrame")

    if df1.shape != df2.shape:
        return False

    num_rows, num_cols = df1.shape
    for i in range(num_rows):
        match = sum(df1.iloc[i] == df2.iloc[i])
        if match != num_cols:
            return False
    return True


class KModesModel:
    def __init__(self, clustroids, cluster_assignment, cluster_score):
        """
        The KModesModel object encapsulates the result of a k-modes cluster run
        :rtype : KModesModel
        :param clustroids: a k*n dataframe (k - num of clusters, n - num of atttrs) indicating the cluster centers
                    (equivalent to centroids in k means)
        :param cluster_assignment: the final cluster_assignment, a nd array of size equal to dataframe.row_size. Each
                   element in the array will have a number between 0 and k-1 indicating the cluster to which the row is
                   assigned
        :param cluster_score: will contain the average distance of each point assigned to a cluster. tells us how far each point
                    in the cluster is to the clustroid. use for computing cluster cohesion
        :return:
        """
        self.clustroids = clustroids
        self.cluster_assignment = cluster_assignment
        self.cluster_score = cluster_score


class KModesClustering:
    def __init__(self, dataframe):
        if not isinstance(dataframe, DataFrame):
            raise Exception(
                "dataframe should be an instance of pandas.DataFrame")
        self.dataframe = dataframe
        (self.num_rows, self.num_attrs) = dataframe.shape

    def cluster(self, k, num_iter):
        """
        the main clustering  method. initializes k clustroids and then runs iterations trying to move
        the clustroids to the center of each cluster while reassigning points to the nearest clustroids

        :type num_iter: int
        :param k: no. of clusters
        :param num_iter: number of iterations to move to the best clustroid
        :return: KModesModel object
        """
        print "k={},num_iter={}".format(k, num_iter)

        clustroids = self.init_random_clustroids(k)
        print "clustroids size after init-{}".format(len(clustroids))
        cluster_attr_freq = [[defaultdict(int) for _ in range(self.num_attrs)]
                             for _ in range(k)]
        cluster_assignment, cluster_score = self.assign_clustroids(clustroids, cluster_attr_freq)
        i = 0
        while num_iter > i:
            prev_clustroids = clustroids
            prev_cluster_assignment = cluster_assignment

            clustroids = self.recompute_clustroids(k, cluster_attr_freq)
            if frames_equal(prev_clustroids, clustroids):
                print "Reached clustroid equilibrium"
                break
            cluster_assignment, cluster_score = self.assign_clustroids(clustroids,
                                                                       cluster_attr_freq, prev_cluster_assignment)
            i += 1
        print "Done clustering i = ", i

        model = KModesModel(clustroids, cluster_assignment, cluster_score)
        return model

    def init_random_clustroids(self, k):
        clustroid_idxs = set([])
        clustroids = []

        while len(clustroid_idxs) < k:
            clustroid_idxs.add(np.random    .randint(0, self.num_rows - 1))

        for clustroid_idx in clustroid_idxs:
            clustroids.append(self.dataframe[self.dataframe.columns].iloc[clustroid_idx, :])

        clustroids = DataFrame(clustroids, columns=self.dataframe.columns)
        return clustroids


    def init_distant_clustroids(self, k):

        clustroids = [self.getRandomRow()]

        while len(clustroids) < k:
            mismatches = self.computeMismatchesWithClustroids(clustroids)
            norm_mismatches = mismatches / mismatches.sum()
            norm_mismatches.sort()

            # get the next random clustroid by finding the next largest of the mismatches
            # above a random value from the current clustroid points
            next_clust_idx = norm_mismatches[norm_mismatches.cumsum() >= np.random.rand()].index[0]
            clustroids.append(self.dataframe[self.dataframe.columns].iloc[next_clust_idx, :])

        clustroids = DataFrame(clustroids, columns=self.dataframe.columns)
        return clustroids

    def assign_clustroids(self, clustroids, cluster_attr_freq, prev_cluster_assignment=None):
        """
        :type prev_cluster_assignment: nd.array
        assignClustroids : to assign each dataframe row  to its nearest cluster(i.e.) the cluster that has the lowest
        mismatch
        :param clustroids: the current clustroids to which we have to assign the dataframe points
        :param cluster_attr_freq: list[list[dict]] stores teh frequency of each category of an attribute in a cluster.
                                  e.g. if cluster 1 has attribute [index 0, col = sex] with cat value 'Female' occuring
                                  15 times, then cluster_attr_freq[1][0]['Female']=15
        :param prev_cluster_assignment: the prev cluster assignments
        :return:
               cluster_assignment :
               cluster_score :

        """
        print "clustroids size in assignClustroids-{}".format(len(clustroids))
        cluster_assignment = []
        cluster_score = np.zeros(clustroids.shape[0])
        cluster_membership_count = np.zeros(clustroids.shape[0])

        for rowIdx, row in self.dataframe.iterrows():

            closest_clustroid, distance = find_closest_clustroid(clustroids, point=row)
            cluster_assignment.append(closest_clustroid)

            # update cluster score and membership count to calculate cohesion
            cluster_score[closest_clustroid] += distance
            cluster_membership_count[closest_clustroid] += 1

            if prev_cluster_assignment is not None:
                if prev_cluster_assignment[rowIdx] == cluster_assignment[rowIdx]:
                    continue  # no need to update cluster_attr_freq if the point did not move to new cluster

                # update cluster_attr_freq for current cluster
            # first convert

            for attr_name, attr_val in row.iteritems():
                # we can cache the attr_idx lookup from name in a dict - TODO
                attr_idx = row.index.get_loc(attr_name)
                # increment the frequency count of the attr vals in the cluster
                cluster_attr_freq[closest_clustroid][attr_idx][attr_val] += 1
                # decrement frequencies of the old cluster where this point was assigned
                if prev_cluster_assignment is not None:
                    cluster_attr_freq[prev_cluster_assignment[rowIdx]][attr_idx][attr_val] -= 1
        cluster_score = cluster_score / cluster_membership_count

        return cluster_assignment, cluster_score

    def recompute_clustroids(self, k, cluster_attr_freq):
        """
        :param k: the num of clusters
        :param cluster_attr_freq: see assign_clustroids() method parameters for defn.
        :return:
                clustroids - the recomputed clustroids
        """
        new_clustroids = np.empty((k, self.num_attrs), dtype='object')

        for clusterIdx in range(k):
            for attrIdx in range(self.num_attrs):
                new_clustroids[clusterIdx, attrIdx] = get_attr_mode(cluster_attr_freq[clusterIdx][attrIdx])

        # replace current clustroids with newClustroids
        clustroids = DataFrame(new_clustroids, columns=self.dataframe.columns)

        return clustroids

    def getRandomRow(self):
        idx = np.random.random_integers(0, self.num_rows - 1)
        return self.dataframe[self.dataframe.columns].iloc[idx, :]

    def computeMismatchesWithClustroids(self, clustroids):
        result = None
        for clustroid in clustroids:
            if result is None:
                result = self.calc_df_mismatch_with_point(clustroid)
            else:
                result = pd.concat(
                    [result, self.calc_df_mismatch_with_point(clustroid)], axis=1).min(axis=1)
        return result

    def calc_df_mismatch_with_point(self, point):
        # return mismatch of data frame with the given point. compares the point with each row in the dataframe and
        # returns a ndarray of mismatch counts in index corresponding to each row comparison
        df_mismatch = pd.Series(index=np.arange(self.num_rows))
        for i in range(self.num_rows):
            df_mismatch[i] = calc_point_mismatches(
                self.dataframe[self.dataframe.columns].iloc[i, :], point)
        return df_mismatch
