
#import preprocessing and model
import preprocessing as pp
import multiple_linear_regression as mlr
import k_means as km
#file name, laptop dataset
file_name = "Laptop_data.csv"

#preprocessing
df, df_clustering, df_regression = pp.preprocessing(file_name)

#K-means Clustering
km.k_means(df,df_clustering)

#Multiple Linear Regression
mlr.multiple_linear_regression(df,df_regression)
