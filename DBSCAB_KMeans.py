import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

if __name__ == '__main__':
    data = pd.read_csv('data/DBSCANdata.txt', sep=' ')
    X = data[['calories', 'sodium', 'alcohol', 'cost']]

    '''
    ============= K-Means 聚类 =======================================
    '''
    km = KMeans(n_clusters=3).fit(X)
    # labels_:每个点的标签. cluster_centers_:集群中心的坐标 类型为： array，[n_clusters，n_features]
    print('每个点标签：', km.labels_)
    print('每个簇中心点：\n', km.cluster_centers_)
    data['cluster'] = km.labels_
    cluster_centers = km.cluster_centers_
    colors = np.array(['red', 'green', 'blue', 'yellow'])
    # 绘制散点图矩阵  输入为 DataFrame 类型
    scatter_matrix(X, alpha=1, s=100, c=colors[data['cluster']], figsize=(8, 5))
    plt.suptitle("With 3 centroids initialized")

    # 标准化数据 通过删除均值和缩放到单位方差来标准化特征
    scaler = StandardScaler()
    # Fit to data, then transform it.
    X_scaled = scaler.fit_transform(X)
    km_scaled = KMeans(n_clusters=3).fit(X_scaled)
    data['scaled_cluster'] = km_scaled.labels_

    # 聚类评估：轮廓系数
    # 特征重要性不一样，所以权重不一样，因此标准化后不一定比原始效果好
    score_scaled = metrics.silhouette_score(X_scaled, labels=data['scaled_cluster'])
    score = metrics.silhouette_score(X, labels=data['cluster'])
    print('标准化后得分：', score_scaled)
    print('未标准化得分：', score)

    # 观察几种分类比较好
    scores = []
    for k in range(2, 20):
        labels_temple = KMeans(n_clusters=k).fit(X).labels_
        score_temple = metrics.silhouette_score(X, labels=labels_temple)
        scores.append(score_temple)

    plt.figure()
    plt.plot(list(range(2, 20)), scores)
    plt.xlabel("Number of Clusters Initialized")
    plt.ylabel("Sihouette Score")
    plt.show()

    '''
    ============== DBSCAN 聚类 ===============================================
    '''
    '''
    eps ： float，可选。两个样本之间的最大距离，以便将它们视为在同一邻域中。相当于半径
    min_samples ： int，可选。对于要被视为核心点的点，邻域中的样本数（或总权重）。这包括点本身。
    fit（X，y =无，sample_weight =无）  从要素或距离矩阵执行DBSCAN聚类。        无返回值。
    fit_predict（X，y =无，sample_weight =无）  在X上执行群集并返回群集标签。   有返回值，是标签
    '''
    dbscan = DBSCAN(eps=10, min_samples=2).fit(X)
    labels_db = dbscan.labels_
    data['cluster_db'] = labels_db
    # DBSCAN 一般不进行均值化
    db_score = metrics.silhouette_score(X, labels=labels_db)
    print('DBSCAN的分数：', db_score)








