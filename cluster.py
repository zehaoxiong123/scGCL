import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle  ##python自带的迭代器模块
def test():
    data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3

    #假如我要构造一个聚类数为3的聚类器
    estimator = KMeans(n_clusters=3)#构造聚类器
    estimator.fit(data)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    centroids = estimator.cluster_centers_ #获取聚类中心
    inertia = estimator.inertia_ # 获取聚类准则的总和

def k_means(data,n_class):
    estimator = KMeans(n_class)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    print(label_pred)
    return label_pred


def SC_cluster(data,n_class):
    metrics_metrix = (-1 * metrics.pairwise.pairwisGaussianMixture_clustere_distances(data)).astype(np.int32)
    metrics_metrix += -1 * metrics_metrix.min()
    ##设置谱聚类函数
    n_clusters_ = n_class
    lables = spectral_clustering(metrics_metrix, n_clusters=n_clusters_)
    print(lables)
    return lables

def GaussianMixture_cluster(data,n_class):
    ##设置gmm函数
    gmm = GaussianMixture(n_components=n_class, covariance_type='spherical').fit(data)
    ##训练数据
    y_pred = gmm.predict(data)
    print(y_pred)
    return y_pred




if __name__=="__main__":
    y_score = np.array([[
        [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
        [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
    ], [
        [0.7, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.56, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.5, 0.3], [0.2, 0.75, 0.12],
        [0.05, 0.05, 0.9], [0.1, 0.8, 0.6], [0.9, 0.8, 0.08],
    ], [
        [0.5, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.7, 0.3],
        [0.2, 0.4, 0.3], [0.1, 0.5, 0.3], [0.2, 0.9, 0.12],
        [0.05, 0.8, 0.9], [0.1, 0.6, 0.6], [0.9, 0.8, 0.08],
    ]])
    print(k_means(y_score,3))