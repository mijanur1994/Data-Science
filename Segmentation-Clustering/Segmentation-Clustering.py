# Import Library 
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# beer dataset
beer = pd.read_csv('beer.txt', sep=' ')
beer


# define X
X = beer.drop('name', axis=1)

from sklearn.cluster import KMeans
get_ipython().run_line_magic('pinfo', 'KMeans')

# K-means with 3 clusters
km = KMeans(n_clusters=3, random_state=1)
km.fit(X)
km.inertia_

# review the cluster labels
km.labels_

# save the cluster labels and sort by cluster
beer['cluster3'] = km.labels_
beer.sort_values('cluster3')


# What do the clusters seem to be based on? Why?
# review the cluster centers
km.cluster_centers_

# calculate the mean of each feature for each cluster
beer.groupby('cluster3').mean()


# save the DataFrame of cluster centers
centers = beer.groupby('cluster3').mean()

# allow plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14


# create a "colors" array for plotting
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])



# scatter plot of calories versus alcohol, colored by cluster (0=red, 1=green, 2=blue)
# cluster centers, marked by "+"
# add labels
plt.scatter(beer.calories, beer.alcohol, c=colors[beer.cluster3], s=50)
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel('calories')
plt.ylabel('alcohol')

# scatter plot matrix (0=red, 1=green, 2=blue)
pd.scatter_matrix(X, c=colors[beer.cluster3], figsize=(10,10), s=100)


# It can be observed that the segments are mostly based on calories. High, medium and low calories. This is because scale of calogies is larger than the scale of other parameters. So, we need to scale all parameters and then cluster it.


####################### Repeat with scaled data###########################
# center and scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pd.DataFrame(X_scaled).describe()


# K-means with 3 clusters on scaled data
km = KMeans(n_clusters=3, random_state=1)
km.fit(X_scaled)

# save the cluster labels and sort by cluster
beer['cluster_3_a'] = km.labels_
beer.sort_values('cluster_3_a')



# What are the "characteristics" of each cluster?
# review the cluster centers
beer.groupby('cluster_3_a').mean()

# scatter plot matrix of new cluster assignments (0=red, 1=green, 2=blue)
pd.scatter_matrix(X, c=colors[beer.cluster_3_a], figsize=(10,10), s=100)

######### Clustering evaluation (Finding optimal number of clusters)############

cmap = sn.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
g = sn.clustermap(X_scaled, cmap=cmap, linewidths=.5)


cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
  clusters = KMeans( num_clusters )
  clusters.fit( X_scaled )
  cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )


clusters_df[0:10]


plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o",color = 'red'  )


