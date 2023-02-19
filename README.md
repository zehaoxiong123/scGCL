# scGCL: an imputation method for scRNA-seq data based on Graph Contrastive Learning

![fram1 (1)](https://github.com/zehaoxiong123/scGCL/blob/main/scGCL.png)

Here, we propose a single-cell Graph Contrastive Learning method for scRNA-seq data imputation, named scGCL, which integrate graph contrastive learning and Zero-inflated Negative Binomial (ZINB) autoencoder to estimate dropout values. scGCL summarizes global and local semantic information through contrastive learning and selects appropriate positive samples to enhance the representation of target nodes. To capture the global probability distribution, scGCL introduces an autoencoder based on the ZINB distribution, which reconstructs the scRNA-seq data based on the prior distribution. Through extensive experiments, we verify that scGCL outperforms existing state-of-the-art imputation methods in clustering performance and gene imputation on 14 real scRNA-seq datasets. Further, we find that scGCL can enhance the expression patterns of specific genes in Alzheimer's disease datasets.

### Requirment

- Python version: 3.8.8
- Pytorch version: 1.10.0
- torch-cluster version: 1.6.0
- torch-geometric version: 2.2.0
- torch-scatter version: 2.0.9
- torch-sparse version: 0.6.13
- faiss-cpu: 1.7.3 

## Usage
You can run the scGCL from the command line:
```
$ python main.py --dataset adam --epochs 300
```
## Arguments
|    Parameter    | Introduction                                                 |
| :-------------: | ------------------------------------------------------------ |
|    dataset     | A h5 file. Contains a matrix of scRNA-seq expression values,true labels, and other information. By default, genes are assumed to be represent-ed by columns and samples are assumed to be represented by rows. |
|  task  | Downstream task. Supported tasks are: node, clustering, similarity                                     |
| es | Early Stopping Criterion                                   |
|     epochs     | Number of training epochs                                    |