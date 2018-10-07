
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
get_ipython().magic('matplotlib inline')


# In[63]:


#Read Data into Dataframe
data = pd.read_csv(r'data\data_stocks.csv')


# In[64]:


#Have a look at the Data
data.head()  


# In[4]:


data.shape # Total Dimension of the Data


# In[5]:


#Keep Original Dataframe and take a copy
dataset = data.copy(deep = True)


# ## Problem 1:
# ## There are various stocks for which we have collected a data set, which all stocks are
# ## apparently similar in performance

# In[6]:


#Drop SP500 column as the Stocks are need to be compared
dataset.drop('SP500', axis=1,inplace=True)


# In[8]:


#Convert the DATE column into Index
dataset.set_index('DATE', inplace=True)


# In[9]:


#For Analyzing the Stocks we need them in rows rather than in column, So transpose them
dataTransposed = dataset.transpose()
dataTransposed.head()


# In[13]:


PCADR = PCA(n_components=3) # Principle Component Analysis - Dimensionality Reduction 
datasetReduced = PCADR.fit_transform(dataTransposed)
datasetReduced.shape


# In[26]:


#Visulaize the Data where the Dimension is Reduced
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.text2D(0.05, 0.95, "Clustering map on Dimension Reduced Data", transform=ax.transAxes)
ax.scatter(datasetReduced[:,0], datasetReduced[:,1], datasetReduced[:,2], c= 'r')


# In[27]:


#Check the Ideal number of Cluster (K) available in the Dataset using Elbow Methods
distortions = []
rng = range(2,20)
for k in rng:
    kmeanModel = KMeans(n_clusters=k).fit(dataTransposed)
    distortions.append(sum(np.min(cdist(dataTransposed, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / dataTransposed.shape[0])


# In[28]:


# Plot the elbow
plt.figure(figsize=(16,9))
plt.plot(rng, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[30]:


#Check the Ideal number of Cluster (K) available in the Dataset using Average silhouette method
for n_clusters in range(2,20):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(dataTransposed)
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dataTransposed, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)


# In[31]:


#From above tests the Elbow test looks more Reliant and no of clusters for this Dataset is 6
KmCluster = KMeans(n_clusters=6) #Create a K Means with 6 Clusters
KmCluster.fit(dataTransposed)


# In[32]:


y_pred = KmCluster.predict(dataTransposed) #Predict to identify to get the Clustering of the Dataset


# In[62]:


#Visualize the Data to see the clustering with Predicted value on the Dimension reduced data
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.text2D(0.05, 0.95, "Clustering map on Dimension Reduced Data", transform=ax.transAxes)
ax.scatter(datasetReduced[:,0], datasetReduced[:,1], datasetReduced[:,2],c=y_pred,marker='x', s=50^2,alpha=0.9 )


# In[53]:


dataTransposed['Clustering'] = y_pred


# In[54]:


#Below number of Clusters available which describes that are performing Similarly
dataTransposed.Clustering.value_counts()


# ## Problem 2:
# ## How many Unique patterns that exist in the historical stock data set, based on
# ## fluctuations in price.

# In[ ]:


#From of Number of Clustering Available There are 6 Patterns which are available in the DataSet, below are the set of Data from
# the 6 patterns


# In[55]:


dataTransposed.loc[dataTransposed['Clustering']==1]


# In[56]:


dataTransposed.loc[dataTransposed['Clustering']==2]


# In[58]:


dataTransposed.loc[dataTransposed['Clustering']==4]


# In[59]:


dataTransposed.loc[dataTransposed['Clustering']==3]


# In[60]:


dataTransposed.loc[dataTransposed['Clustering']==5]


# In[61]:


dataTransposed.loc[dataTransposed['Clustering']==0]


# ## Problem 3:
# ## Identify which all stocks are moving together and which all stocks are different from
# ## each other.

# In[ ]:


#To find the correlation lets use the corr function available in pandas dataframe
#Correlation has to be done for the  stock returns so use pct_change of pandas


# In[34]:


dataset.head()


# In[35]:


#In finance we calculate correlations between stock returns and not stock prices,as returns tend 
#to follow normal distribution and prices don't
dataCorr = dataset.pct_change().corr()  #Percent Change and the Correlation function
dataCorr.head()


# In[42]:


dataCorr.head(50) # Sample Output of the First 50 Records
#Records Which are close to +1 and -1 are closely Related


# In[41]:



#Function to Create the Heat Map Function
def VisualizeWithHeadMap(dataCorrSubset):
    '''
    Function which creates an HeatMap Graph for a datafram
    Parm : Dataframe
    '''
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    heatmap = ax.pcolor(dataCorrSubset, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(dataCorrSubset.shape[0] + 0.5))
    ax.set_yticks(np.arange(dataCorrSubset.shape[0] + 0.5))
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    columns = dataCorrSubset.columns
    rows = dataCorrSubset.index

    ax.set_xticklabels(columns)
    ax.set_yticklabels(rows)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()


# In[44]:


#Heat Map of First 20 Records
VisualizeWithHeadMap(dataCorr.iloc[:20,:20])



# In[45]:


#Heat Map of  20  to 40 Records
VisualizeWithHeadMap(dataCorr.iloc[20:40,20:40])


# In[46]:


#Heat Map of  40  to 60 Records
VisualizeWithHeadMap(dataCorr.iloc[40:60,40:60])


# In[47]:


#Heat Map of  60  to 80 Records
VisualizeWithHeadMap(dataCorr.iloc[60:80,60:80])


# In[49]:


#Heat Map of  80  to 100 Records
VisualizeWithHeadMap(dataCorr.iloc[80:100,80:100])

