# scGCL

![image](https://github.com/zehaoxiong123/scGCL/blob/main/scGCL.png)

scGCL: Here, we propose a single-cell Graph Contrastive Learning method for scRNA-seq data imputation, named scGCL, which integrate graph contrastive learning and Zero-inflated Negative Binomial (ZINB) autoencoder to estimate dropout values. scGCL summarizes global and local semantic information through contrastive learning and selects appropriate positive samples to enhance the representation of target nodes. To capture the global probability distribution, scGCL introduces an autoencoder based on the ZINB distribution, which reconstructs the scRNA-seq data based on the prior distribution. Through extensive experiments, we verify that scGCL outperforms existing state-of-the-art imputation methods in clustering performance and gene imputation on 14 real scRNA-seq datasets. Further, we find that scGCL can enhance the expression patterns of specific genes in Alzheimer's disease datasets.
