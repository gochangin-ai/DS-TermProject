import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means(df, x):
    # PCA
    pca = PCA(n_components=1)
    pca_x = pca.fit_transform(x)
    pca_new_df = pd.DataFrame(data=pca_x, columns=['pca'])
    pca_new_data = pca_new_df.values

    plt.scatter(df['latest_price'].values, pca_new_data)

    # KMeans Clustering
    clust_df = pca_new_df.copy()
    clust_df['price'] = df['latest_price'].reset_index(drop=True)
    # standard.fit_transform(df['latest_price'].values.reshape(-1,1))

    KM = KMeans(n_clusters=4)
    KM.fit(clust_df)

    centers = KM.cluster_centers_
    pred = KM.predict(clust_df)
    clust_df['clust'] = pred

    targets = [0, 1, 2, 3]
    colors = ['r', 'g', 'b', 'pink']

    for target, color in zip(targets, colors):
        indicesToKeep = clust_df['clust'] == target
        plt.scatter(clust_df.loc[indicesToKeep, 'price'], clust_df.loc[indicesToKeep, 'pca'], c=color, s=10)
    plt.scatter(centers[:, 1], centers[:, 0], c='black', s=30, marker='s')
    plt.xlabel('Price')
    plt.ylabel('Component PCA Value')
    plt.title("Full Data")
    plt.show()