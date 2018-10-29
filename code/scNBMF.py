'''
Code for the paper scNBMF: Single-Cell Negative Binomial Matrix Factorization from Single Cell RNAseq Data

'''

from time import time

import click
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import Birch
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score


tf.logging.set_verbosity(tf.logging.INFO)


def Datapreprocess(data,obs_col,var_col,val_col,offset_col,verbose):
    '''
    Proprocess for raw input data

    Notes:
    It will get the total count of each genes for training and transform the data into 4 columns 

    Variables:
    data: The input raw data  (shape: genes x cell)
    obs_col: The index of the column in result data which contains the cell index of the raw data
    obs_col: The index of the column in result data which contains the gene index of the raw data
    obs_col: The index of the column in result data which contains the count expression of the raw data
    obs_col: The index of the column in result data which contains the total count of the raw data
    verbose: Whether the dimensional information need to be output or not
    '''
    if verbose:
        print "Datapreprocess..."
        print "Data rows %d " %data.shape[0]
        print "Data cols %d " %data.shape[1]
        print "obs_col %d " %obs_col
        print "var_col %d " %var_col
        print "val_col %d " %val_col
        print "offset_col %d " %offset_col
    
    res = np.zeros((data.shape[0] * data.shape[1], 4),dtype = 'int32')
    colsum = np.zeros(data.shape[1],dtype = 'int32')
    for j in range(data.shape[1]):
        colsum[j] = 0
        for i in range(data.shape[0]):
            colsum[j] = colsum[j] + data[i,j]

    t = 0
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            res[t,obs_col] = j 
            res[t,var_col] = i
            res[t,val_col] = int(data[i,j])
            res[t,offset_col] = int(colsum[j])
            t = t + 1

    #np.savetxt("./data2", res, delimiter = ',', fmt = '%d') 
    return pd.DataFrame(data = res)


def get_weight(W,lambda1):
    '''
    Function to get the l1_penalty of W

    Variables:
    W: The matrix to add l1_penalty
    lambda1: The coeffcient of l1_penalty
    res = lambda1 * (|w1| + |w2| + ... + |wn|)
    '''
    res = W
    tf.add_to_collection('collection', tf.contrib.layers.l1_regularizer(lambda1)(res))  
    return res


def get_weight2(W,lambda1):
    '''
    Function to get the l2_penalty of W

    Variables:
    W: The matrix to add l2_penalty
    lambda1: The coeffcient of l2_penalty
    res = lambda1 * ((w1)^2 + (w2)^2 + ... + (wn)^2)
    '''
    res = W
    tf.add_to_collection('collection', tf.contrib.layers.l2_regularizer(lambda1)(res))  
    return res


def next_batch(data, batch_size, i, NN):
    '''
    Function to get the next batch of the training data

    Notes:
    The data will be used recyclable

    Variables:
    data: The input data with preprocessing (shape: genes*cells x 4)
    batch_size: The training size of the batch
    i : The number of the iterations
    NN: The whole rows of data(genes*cells)
    '''
    indx = (batch_size * i) % NN
    if (batch_size + indx) > NN:
        indx = 1

    return data.iloc[indx:indx + batch_size]


def CalculateCluster(H_result,label_true,f,t0,nmi_record,times=100,clusters=8):
    '''
    8 kinds of cluster methods

    Notes: Calculating several times of clusters and get the median as the final result
    H_result: The final result of dimensional reduction (shape: cells * k)
    label_true: The true label
    f: The output file to record the cluster result of NMI AMI and ARI
    times: The times of clusters
    nmi_record: Record the scores of Kmeans NMI to print it on the screen
    clusters: The kinds which need to be divided into
    '''
    
    kmeans(H_result,label_true,f,times,clusters,nmi_record)
    minibatchkmeans(H_result,label_true,f,times,clusters)
    spectralclustering_kmeans(H_result,label_true,f,times,clusters)
    spectralclustering_discretize(H_result,label_true,f,times,clusters)
    agglomerativeclustering_ward(H_result,label_true,f,times,clusters)
    agglomerativeclustering_complete(H_result,label_true,f,times,clusters)
    agglomerativeclustering_average(H_result,label_true,f,times,clusters)
    birch(H_result,label_true,f,times,clusters)
    print >> f , "time2\t%.2f" %(time() - t0)


def getevaluate(label_true,label_pred,nmi,ami,ari):
    '''
    Get the NMI AMI and ARI result of each label we predict.

    Notes:
    The NMI AMI and ARI result will be listed
    '''
    nmi.append(normalized_mutual_info_score(label_true, label_pred))
    ami.append(adjusted_mutual_info_score(label_true, label_pred))
    ari.append(adjusted_rand_score(label_true, label_pred))


def getmedian(nmi,ami,ari):
    '''
    Get the median of each list
    '''
    nmi_result = np.median(nmi)
    ami_result = np.median(ami)
    ari_result = np.median(ari) 
    return nmi_result,ami_result,ari_result



def kmeans(H_result,label_true,f,times,clusters,nmi_record):
    '''
    Kmeans cluster methods

    Notes: Calculating several times of kmeans and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = KMeans(n_clusters = clusters)
        estimator.fit(H_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result,ami_result,ari_result =  getmedian(nmi,ami,ari)
    nmi_record = nmi_result
    print >> f , "KMeans\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def minibatchkmeans(X_result,label_true,f,times,clusters):
    '''
    MiniBatchKmeans cluster methods

    Notes: Calculating several times of MiniBatchKmeans and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = MiniBatchKMeans(n_clusters = clusters)
        estimator.fit(X_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)

    print >> f , "MiniBatchKMeans\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def spectralclustering_kmeans(X_result,label_true,f,times,clusters):
    '''
    Spectral Clustering methods

    Notes: Calculating several times of Spectral Clustering and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = SpectralClustering(n_clusters = clusters,assign_labels='kmeans')
        estimator.fit(X_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)
    print >> f , "SpectralClustering_kmeans\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def spectralclustering_discretize(X_result,label_true,f,times,clusters):
    '''
    Spectral Clustering methods

    Notes: Calculating several times of Spectral Clustering and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = SpectralClustering(n_clusters = clusters,assign_labels='discretize')
        estimator.fit(X_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)
    print >> f , "SpectralClustering_discretize\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def agglomerativeclustering_ward(X_result,label_true,f,times,clusters):
    '''
    Agglomerative Clustering methods

    Notes: Calculating several times of Agglomerative Clustering methods and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = AgglomerativeClustering(n_clusters = clusters)
        estimator.fit(X_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)
    print >> f , "AgglomerativeClustering_ward\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def agglomerativeclustering_complete(X_result,label_true,f,times,clusters):
    '''
    Agglomerative Clustering methods

    Notes: Calculating several times of Agglomerative Clustering methods and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = AgglomerativeClustering(n_clusters = clusters,linkage='complete')
        estimator.fit(X_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)
    print >> f , "AgglomerativeClustering_complete\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def agglomerativeclustering_average(X_result,label_true,f,times,clusters):
    '''
    Agglomerative Clustering methods

    Notes: Calculating several times of Agglomerative Clustering methods and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        estimator = AgglomerativeClustering(n_clusters = clusters,linkage='average')
        estimator.fit(X_result)
        label_pred = estimator.labels_.tolist()
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)
    print >> f , "AgglomerativeClustering_average\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def birch(X_result,label_true,f,times,clusters):
    '''
    Birch Clustering methods

    Notes: Calculating several times of Birch and get the median as the final result
    '''
    nmi = []
    ami = []
    ari = []

    for i in range(times):
        label_pred = Birch(n_clusters = clusters).fit_predict(X_result)
        getevaluate(label_true,label_pred,nmi,ami,ari)

    nmi_result, ami_result, ari_result = getmedian(nmi,ami,ari)
    print >> f , "Birch\tnmi%.6f\tami%.6f\tari%.6f\t" %(nmi_result, ami_result, ari_result),


def Pyplotshow(H_result,costs,nmi_result,C,G,title,t0):
    '''
    Paint the result

    Notes: There are two figures. Contains the first two dimension of the result and the cost change.
    H_result: The final result of dimensional reduction (shape: cells * k)
    costs: The training loss funcion
    nmi_record: Record the scores of Kmeans NMI to print it on the screen
    C: Number of cells
    G: Number of genes
    title: The title of the figures
    t0: The start time of running the project
    '''

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(H_result[0], H_result[1], s=10, alpha=0.33, c='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title + ' {} X {}'.format(C, G))

    plt.subplot(1, 2, 2)
    plt.plot(costs, c='k')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')

    if nmi_result > 0:
        plt.title('Runtime: {:.2f}s NMI: {:.6f}'.format((time() - t0),nmi_result))
    else:
        plt.title('Runtime: {:.2f}s '.format(time() - t0))
    plt.tight_layout()
    plt.show()



def scNBMF_model(G, C, k, variable_idx, sample_idx, T_, y_, psi, penalty_type, lambda_for_l1, eps = 1e-8):
    '''
    scNBMF model
    
    G: Number of genes
    C: Number of cells
    variable_idx: Gene index in the raw data matrix
    sample_idx: Cell index in the raw data matrix
    T_: Total count in the raw data matrix
    y_: Count expression in the raw data matrix
    psi: EdgeR dispersion of the input count data
    penalty_type: 1 means l1_penalty and others means l2_penalty
    lambda_for_l1: The coeffcient of l1 or l2_penalty

    return:
    LL : loss function for the model
    '''
    W = tf.Variable(np.random.randn(G, k), name='weights')

    H = tf.Variable(np.random.randn(k, C), name='PCs')

    S = tf.Variable(np.array([0.]), name='Scaling')

    W_ = tf.gather(W, variable_idx)
    psi_ = tf.gather(psi, variable_idx)

    H_ = tf.gather(tf.matrix_transpose(H), sample_idx)
    eta_ = tf.reduce_sum(W_ * H_, 1)

    mu_ = tf.exp(eta_ + S + tf.log(T_))

    LL = tf.reduce_sum(y_ * tf.log(mu_ + eps) - (y_ + psi_) * tf.log(mu_ + psi_ + eps)) 

    if penalty_type == 1:
        Wpenalty = get_weight(W ,lambda_for_l1)
    else:
        Wpenalty = get_weight2(W ,lambda_for_l1)

    beta = 1;
    LL = tf.reduce_mean(LL + beta * Wpenalty)

    return LL



## Model ##
##Example
## sudo python scNBMF.py brainTags.filtered.counts.withoutNames.txt
## sudo python scNBMF.py --psi_file trend.disp.txt brainTags.filtered.counts.withoutNames.txt 
## sudo python scNBMF.py --calcluster True --tagsname brainTags_cellType.txt --cluster_num 8 --psi_file trend.disp.txt brainTags.filtered.counts.withoutNames.txt
## sudo python scNBMF.py --calcluster True --tagsname brainTags_cellType.txt --cluster_num 8 --psi_file trend.disp.txt --num_iter 18000 --verbose True brainTags.filtered.counts.withoutNames.txt

@click.command()
@click.argument('input_file')
@click.option('--calcluster', default=False)
@click.option('--cluster_num', default=8)
@click.option('--tagsname', default='')#important 
@click.option('--batch_size', default=10000)
@click.option('--num_iter', default=18000)
@click.option('--learning_rate', default=0.001)
@click.option('--inner_iter', default=5)
@click.option('--report_every', default=100)
@click.option('--ndim', default=20)
@click.option('--penalty_type', default=1)
@click.option('--lambda_for_l1', default=0.3)
#@click.option('--psi', default=1)
@click.option('--storename', default="./H_result.txt")
@click.option('--result_file', default="./cluster_result.txt")
@click.option('--psi_file', default="./trend.disp.txt")
@click.option('--pyplot', default=True)
@click.option('--title', default='')
@click.option('--verbose', default=False)


def main(input_file, calcluster, cluster_num, tagsname, batch_size,
         num_iter, learning_rate, inner_iter, report_every, ndim, penalty_type, 
         storename, result_file ,psi_file ,lambda_for_l1, pyplot, title, verbose):
    '''
    Main parametrization of the scNBMF algorithm.

    Notes: We recommend to calculate psi before run this project and add --psi_file name_of_the_psifile.txt
    +If you need cluster after calculate the dimensional reduction matrix you need to add --calcluster True --tagsname name_of_the_label.txt --cluster_num the_number_of_clusters
    
    input_file: The input count expression matrix(shape: genes x cells)
    calcluster: Boolean variable. Whether need to calculate clusters or not
    cluster_num: The kinds which need to be divided into
    tagsname: The name of the label file
    batch_size: Size of the training batch
    num_iter: Iterations of training process
    learning_rate; Learning rate of the adam optimizer
    inner_iter: Calculate how many times of optimizer before refresh loss
    report_every: Show tf.logging.INFO when training several iterations
    ndim: (k) The numbers of final dimension need to be reducted
    penalty_type: 1 means l1_penalty and others means l2_penalty
    storename: The output file to record the dimensional reduction matrix
    result_file: The output file to record the cluster result of NMI AMI and ARI
    psi_file: File name of the edgeR dispersion of the input count data
    lambda_for_l1: The coeffcient of l1 or l2_penalty
    pyplot: Boolean variable. Whether need to paint figures
    title: The title of the figures
    verbose: Show verbose infprmation
    '''

    ## Data loading ##
    f = open(result_file, "a+")
    data = np.loadtxt(input_file,dtype = 'int32', delimiter = "\t")
    if data is None:
        raise ValueError("Cannot load the count data expression file")

    psi = np.loadtxt(psi_file,dtype = 'float64', delimiter = "\n")
    if psi is None:
        raise ValueError("Cannot load the dispersion file")

    obs_col = 0
    var_col = 1
    val_col = 2
    offset_col = 3

    data = Datapreprocess(data,obs_col,var_col,val_col,offset_col,verbose)

    if verbose:
        print psi
        print data.head()

    ## CONFIG ##
    G = data[var_col].unique().shape[0]
    C = data[obs_col].unique().shape[0]

    if verbose:
        print "The number of genes %d" %G
        print "The number of genes %d" %C
    t0 = time()
    tf.logging.info('Shuffling...')
    data = data.sample(frac=1)
    NN = data.shape[0]

    k = ndim
    
    ## Model ##
    sample_idx = tf.placeholder(tf.int32, shape=[None])# Gene index in the raw data matrix
    variable_idx = tf.placeholder(tf.int32, shape=[None])# Cell index in the raw data matrix
    T_ = tf.placeholder(tf.float64, shape=[None])# Total count in the raw data matrix
    y_ = tf.placeholder(tf.float64, shape=[None])# Count expression in the raw data matrix

    LL = scNBMF_model(G, C, k, variable_idx, sample_idx, T_, y_, psi, penalty_type, lambda_for_l1)

    cost = -LL / batch_size

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
                            
    init = tf.global_variables_initializer()

    costs = np.zeros(num_iter)

    if verbose:
        tf.logging.info('Training')
    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_iter):
            batch = next_batch(data, batch_size, i, NN)
            feed_dict = {sample_idx: batch[obs_col],
                        variable_idx: batch[var_col],
                        y_: batch[val_col],
                        T_: batch[offset_col]}

            for j in range(inner_iter):
                sess.run(optimizer, feed_dict=feed_dict)

            c = sess.run(cost, feed_dict=feed_dict)
            costs[i] = c

            if not i % report_every:
                tf.logging.info('Cost: {}'.format(c))
        
        H_result = sess.run(H)
        np.savetxt(storename, H_result, delimiter = '\t' , fmt = '%.4f') 
        H_result = tf.matrix_transpose(H_result)   
        H_result = H_result.eval()

    if verbose: 
        if calcluster:
            print "Need to calculate cluster..."
        else:
            print "Don't need to calculate cluster...."
        print "Tagsname is %s" %tagsname

    nmi_record = 0
    if calcluster:
        if tagsname == '':
            raise ValueError("Need tags name(label of the cell) to calculate the result of NMI and ARI")
        else:
            label_true = np.loadtxt(tagsname, dtype='string')
            CalculateCluster(H_result,label_true,f,t0,nmi_record,times=100,clusters=cluster_num)   

    if pyplot:
        H_result = np.loadtxt(storename, delimiter = "\t")
        Pyplotshow(H_result,costs,nmi_record,C,G,title,t0)
        

if __name__ == '__main__':
    main()
