
#import preprocessing and model
import preprocessing as pp
import multiple_linear_regression as mlr
import k_means as km
#file name, laptop dataset
file_name = "Laptop_data.csv"

#preprocessing
df, df_clustering, df_regression = pp.preprocessing(file_name)

#K-means Clustering
clust_df,centers = km.k_means(df,df_clustering)
km.Clustering_visualize(clust_df,centers)
pca_rank_first, pca_rank_second, pca_rank_third, pca_rank_fourth = km.Clustering_Info(clust_df)
print("PCA rank first to forth")
print(pca_rank_first)
print(pca_rank_second)
print(pca_rank_third)
print(pca_rank_fourth)

#Multiple Linear Regression
mlr.multiple_linear_regression(df,df_regression)
