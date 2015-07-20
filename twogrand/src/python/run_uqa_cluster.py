import pandas as pd
import os as os
import datetime
import time
import numpy as np
from k_modes_clustering import KModesClustering


def run_cluster(input_file="/data/twogrand/input/user_question_answer_pivot.csv",
        header=0,
        max_k=100,
        k_steps=5,
        num_runs=3,
        num_iterations=20,
        output_dir="/data/twogrand/output",):

    """
    the main run cluster method that takes in an input csv of user_question_answer and runs kmodes clustering against it
    Writes the clustering result along with its scores to the output dir for each k value
    :param input_file: the input csv file path
    :param header: int. indicates the header line (0-based). header=0 means first line of file will be treated as df
                  header
    :param max_k: the max value of k, max_k should not exceed half the dataframe row size, if a larger value is given,
                 the code will automatically set to half the data size
    :param k_steps: increments of k
    :param num_runs: num of runs within a 'k' clustering. We do this to run with different initial clustroids and choose
                    the one with the least distance to the centroids
    :param num_iterations:number of iteration within a single k run
    :param output_dir: the output folder where we store the results.
    :return:
    """
    uqa_df = pd.read_csv(input_file,header)
    cluster_df(uqa_df,max_k,k_steps,num_iterations,output_dir)


def cluster_df(uqa_df,
        max_k=100,
        k_steps=5,
        num_runs=3,
        num_iterations=20,
        output_dir="/data/twogrand/output"):

    qa_df = uqa_df.ix[:,1:]  # skip the userid column as we want the clustering to be done on the qa points

    num_rows = qa_df.shape[0]
    if max_k > num_rows/2 :
        max_k = num_rows/2

    today = datetime.date.today()
    output_dir = output_dir + "/" +`today.year`+`today.month`+`today.day`

    curtime = time.time()
    output_dir = output_dir+"/run_"+`curtime`

    km = KModesClustering(qa_df)

    k_scores = np.
    for k in range(2, max_k, k_steps) :

        best_model = None
        runs = []
        for i in range (num_runs):
            model = km.cluster(k,num_iterations)
            runs.append(np.nansum(model.cluster_score))

            if best_model == None :
                best_model = model
            else :
                if np.nansum(model.cluster_score) < np.nansum(best_model.cluster_score):
                    best_model = model  # we want the cluster that has low cluster_score(low num mismatches)

        user_cluster= pd.concat([uqa_df.ix[:,0], pd.DataFrame(best_model.cluster_assignment)],axis=1)
        user_cluster.columns=["user_id","cluster"]

        k_dir = output_dir+"/k_"+`k`
        os.makedirs(k_dir,mode=0o744)
        best_model.clustroids.to_csv(k_dir + "/clustroids.csv",index=False)
        user_cluster.to_csv(k_dir + "/user_cluster.csv",index=False)
        pd.DataFrame(runs).to_csv(k_dir + "/run_scores.csv", index=False)


