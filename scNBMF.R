###############################################
####Code for the paper A fast and efficient count-based matrix factorization method for detecting cell types from single-cell RNAseq data
###############################################


library(tensorflow)
library(edgeR)
library(igragh)
library(Rtsne)

tf$logging$set_verbosity(tf$logging$INFO)
np <- import("numpy")

Datapreprocess <- function(data,obs_col,var_col,val_col,offset_col,verbose,row ,col ) {
#################################
    #Proprocess for raw input data
    #
    #Notes:
    #It will get the total count of each genes for training and transform the data into 4 columns 
    #
    #Variables:
    #data: The input raw data  (shape: genes x cell)
    #obs_col: The index of the column in result data which contains the cell index of the raw data
    #obs_col: The index of the column in result data which contains the gene index of the raw data
    #obs_col: The index of the column in result data which contains the count expression of the raw data
    #obs_col: The index of the column in result data which contains the total count of the raw data
    #verbose: Whether the dimensional information need to be output or not
#################################
    if (verbose) {
        cat("Datapreprocess...\n");
        cat( sprintf("Data rows %d \n", nrow(data)) );
        cat( sprintf("Data cols %d \n", ncol(data)) );
        cat( sprintf("obs_col %d \n" , obs_col) );
        cat( sprintf("var_col %d \n" , var_col) );
        cat( sprintf("val_col %d \n" , val_col) );
        cat( sprintf("offset_col %d \n" , offset_col) );
    }

    X <- matrix(0:0,ncol=4,nrow=col*row);
    
    colsum <- colSums(data);
    t <- 1;
    for (i in 1 : col){
        for(j in 1 : row){
            #X[i*j,1] = i*j -1;
            X[t,obs_col] <- i - 1;
            X[t,var_col] <- j - 1;
            X[t,val_col] <- data[j,i];
            X[t,offset_col] <- colsum[i];
            t <- t+1;
        }
    }  

    return(X)
}

get_weight <- function(W,lambda1){
####################################
    #Function to get the l1_penalty of W
    #
    #Variables:
    #W: The matrix to add l1_penalty
    #lambda1: The coeffcient of l1_penalty
    #res = lambda1 * (|w1| + |w2| + ... + |wn|)
####################################
    tmpW <- W
    tf$add_to_collection("collection",tf$contrib$layers$l1_regularizer(lambda1)(tmpW))
    Wpenalty <- tmpW
    return(Wpenalty)
}
    

get_weight2 <- function(W,lambda1){
####################################
    #Function to get the l2_penalty of W
    #
    #Variables:
    #W: The matrix to add l2_penalty
    #lambda1: The coeffcient of l2_penalty
    #res = lambda1 * ((w1)^2 + (w2)^2 + ... + (wn)^2)
####################################
    tmpW <- W
    tf$add_to_collection("collection",tf$contrib$layers$l2_regularizer(lambda1)(tmpW))
    Wpenalty <- tmpW
    return(Wpenalty)
}

next_batch <- function(data, batch_size, i, NN){
###################################
    #Function to get the next batch of the training data
    #
    #Notes:
    #The data will be used recyclable
    #
    #Variables:
    #data: The input data with preprocessing (shape: genes*cells x 4)
    #batch_size: The training size of the batch
    #i : The number of the iterations
    #NN: The whole rows of data(genes*cells)
####################################
    indx <- (batch_size * i) %% NN + 1
    if ((batch_size + indx) > NN){
        indx <- 1
    }

    return(data[indx: (indx+batch_size -1 ) , ])
}
    

scNBMF_model <- function(H, G, C, k, variable_idx, sample_idx, T_, y_, psi, penalty_type, lambda_for_l1, eps = 1e-8){
###################################
    #ZINBMF model
    #
    #H: Res Matrix of dimensional reduction
    #G: Number of genes
    #C: Number of cells
    #variable_idx: Gene index in the raw data matrix
    #sample_idx: Cell index in the raw data matrix
    #T_: Total count in the raw data matrix
    #y_: Count expression in the raw data matrix
    #psi: EdgeR dispersion of the input count data
    #penalty_type: 1 means l1_penalty and others means l2_penalty
    #lambda_for_l1: The coeffcient of l1 or l2_penalty
    #
    #return:
    #LL : loss function for the model
#####################################
    W <- tf$Variable(tf$random_normal(shape(G, k)), name='weights')

    #S <- tf$Variable(tf$zeros(shape(1L)))

    W_ <- tf$gather(W, variable_idx)
    psi_ <- tf$gather(psi, variable_idx)

    H_ <- tf$gather(tf$matrix_transpose(H), sample_idx)
    eta_ <- tf$reduce_sum(W_ * H_, 1L)

    #mu_ <- tf$exp(eta_ + S + tf$log(T_))
    mu_ <- tf$exp(eta_ + tf$log(T_))

    LL <- tf$reduce_sum(y_ * tf$log(mu_ + eps) - (y_ + psi_) * tf$log(mu_ + psi_ + eps)) 

    if (penalty_type == 1){
        Wpenalty <- get_weight(W ,lambda_for_l1)
    }
    else{
        Wpenalty <- get_weight2(W ,lambda_for_l1)
    }

    LL <- tf$reduce_mean(LL + Wpenalty)

    return(LL)

}
   
fill_feed_dict <- function(data, batch_size, i, NN, 
                            obs_col, var_col, val_col, offset_col,
                            sample_idx, variable_idx, y_, T_) {
  # Create the feed_dict for the placeholders filled with the next
  # batch size` examples.
  batch <- next_batch(data, batch_size, i, NN)
  sample_feed <- batch[,obs_col]
  variable_feed <- batch[,var_col]
  y_feed <- batch[,val_col]
  T_feed <- batch[,offset_col]
  dict(
    sample_idx = sample_feed, 
    variable_idx = variable_feed, 
    y_ = y_feed, 
    T_ = T_feed 
  )
}


CalculateCluster <- function(H_result,label_true,result_file,t0,num_clusters,times=100){

    #several kinds of cluster methods

    #Notes: Using several kinds of methods to  calculate several times of clusters and get the median as the final result
    #H_result: The final result of dimensional reduction (shape: cells * k)
    #label_true: The true label
    #f: The output file to record the cluster result of NMI AMI and ARI
    #times: The times of clusters
    #nmi_record: Record the scores of Kmeans NMI to print it on the screen
    #num_clusters: The kinds which need to be divided into

    nmi <- c();
    ari <- c();
     for(j in 1 : 100){
        res_clust <- kmeans(H_result, num_clusters);
            
        nmi <- c(nmi, compare(unlist(res_clust$cluster), unlist(label_true), method = "nmi") );
        ari <- c(ari, compare(unlist(res_clust$cluster), unlist(label_true), method = "adjusted.rand") );
    }
    nmi <- median(nmi);
    ari <- median(ari);
    
    cat("Kmeans\n",file = result_file);
    cat( sprintf("NMI %f ARI %f\n" , nmi, ari) ,file = result_file , append = TRUE);

    cat("Kmeans\n");
    cat( sprintf("NMI %f ARI %f\n" , nmi, ari) );
}

scNBMF <- function(input_file, calcluster = FALSE, tagsname = '', batch_size = 10000,
         num_iter = 18000, learning_rate = 0.001, inner_iter = 5, report_every = 100, ndim = 20, penalty_type = 1, 
         storename = "./H_result.csv", result_file = "./cluster_result.txt", lambda_for_l1 = 0.3, 
         tsneshow = TRUE, title = '', verbose = TRUE) {
#############################
    #Main parametrization of the ZINBMF algorithm.

    #Notes: If you need cluster after calculate the dimensional reduction matrix you need to add --calcluster True --tagsname name_of_the_label.txt 
    
    #input_file: The input count expression matrix(shape: genes x cells)
    #calcluster: Boolean variable. Whether need to calculate clusters or not
    #tagsname: The name of the label file
    #batch_size: Size of the training batch
    #num_iter: Iterations of training process
    #learning_rate; Learning rate of the adam optimizer
    #inner_iter: Calculate how many times of optimizer before refresh loss
    #report_every: Show tf.logging.INFO when training several iterations
    #ndim: (k) The numbers of final dimension need to be reducted
    #penalty_type: 1 means l1_penalty and others means l2_penalty
    #storename: The output file to record the dimensional reduction matrix
    #result_file: The output file to record the cluster result of NMI AMI and ARI
    #psi_file: File name of the edgeR dispersion of the input count data
    #lambda_for_l1: The coeffcient of l1 or l2_penalty
    #pyplot: Boolean variable. Whether need to paint figures
    #title: The title of the figures
    #verbose: Show verbose infprmation
############################
    
    data_input <- read.table(input_file,sep = ',');
    #label_input <- read.table(tagsname);

    data <- data_input[rowSums(data_input)>10,]
    col <- as.integer(ncol(data));
    row <- as.integer(nrow(data));

    obs_col <- 1L
    var_col <- 2L
    val_col <- 3L
    offset_col <- 4L

    G <- row
    C <- col

    edgeres <- DGEList(counts = data)
    edgeres <- estimateDisp(edgeres, robust = TRUE)

    psi <- edgeres$tagwise.dispersion
    data <- Datapreprocess(data,obs_col,var_col,val_col,offset_col,verbose,row,col)

    NN <- as.integer(nrow(data))
    if (verbose){
        cat (psi)
        cat ( sprintf("\n"));
        cat ( sprintf("The number of genes %d\n" , G) );
        cat ( sprintf("The number of genes %d\n" , C) );
    }

    data <- data[sample(nrow(data),nrow(data),replace=FALSE),] #Reshuffle

    t0 <- Sys.time();

    with(tf$Graph()$as_default(), {

        k <- as.integer(ndim);

        H <- tf$Variable(tf$random_normal(shape(k, C)), name='PCs');#Res matrix #np.random.randn(G, k)

        sample_idx <- tf$placeholder(tf$int32, shape(batch_size) );# Gene index in the raw data matrix
        variable_idx <- tf$placeholder(tf$int32, shape(batch_size) );# Cell index in the raw data matrix
        T_ <- tf$placeholder(tf$float32, shape(batch_size) );# Total count in the raw data matrix
        y_ <- tf$placeholder(tf$float32, shape(batch_size) );# Count expression in the raw data matrix

        LL <- scNBMF_model(H, G, C, k, variable_idx, sample_idx, T_, y_, psi, penalty_type, lambda_for_l1)

        cost <- -LL / batch_size

        optimizer <- tf$train$AdamOptimizer(learning_rate)$minimize(cost)
                                
        init <- tf$global_variables_initializer()

        costs <- tf$zeros(shape(num_iter))

        if (verbose){
            cat('Training\n')
        }

        sess <- tf$Session()
        
        sess$run(init)

        for (i in 0:num_iter){
            
            feed_dict <- fill_feed_dict(data, batch_size, i, NN, 
                                        obs_col, var_col, val_col, offset_col,
                                        sample_idx, 
                                        variable_idx, 
                                        y_, 
                                        T_) 

            for (j in 0:inner_iter){
                sess$run(optimizer, feed_dict=feed_dict)
            }

            c <- sess$run(cost, feed_dict=feed_dict)
            #costs[i] <- c

            if( i %% report_every == 0){
                cat(sprintf('Cost: %f \n', c))
            }
        }
        H_result <- sess$run(H)
        #np.savetxt(storename, H_result, delimiter = '\t' , fmt = '%.4f') 
        #H_result = tf$matrix_transpose(H_result)   
        #H_result = H_result$eval()
    })

    #return (H_result)
    H_result <- t(H_result)
    save_H_result <- round(H_result,6);
    #cat (save_H_result)
    write.csv(save_H_result,file = storename);    

    if(verbose){
        if(calcluster){
            cat(sprintf("Need to calculate cluster...\n"))
        }
        else{
            cat(sprintf("Don't need to calculate cluster....\n"))
        }
        cat(sprintf("Tagsname is %s\n" ,tagsname))
    }
    
    if(calcluster){
        if(tagsname == ''){
            cat(sprintf("Need tags name(label of the cell) to calculate the result of NMI and ARI"))
        }
        else{
            label_true <- read.table(tagsname)
            num_clusters <- nrow(unique(label_true))
            #cat (num_clusters)
            CalculateCluster(H_result,label_true,result_file,t0,num_clusters = num_clusters,times = 100)
        }
    }
    
    if(tsneshow){
        res_tsne <- Rtsne(H_result)
        plot(res_tsne$Y, col=label_true$V1, xlab="tSNE1", ylab="tSNE2",cex.axis=1.5,cex.lab=1.3)
        title("scNBMF tSNE, 2 Groups with Labels")
    }

}


