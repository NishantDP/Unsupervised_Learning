# Unsupervised_Learning
# Basic implementation of some of the Unsupervised machine learning techniques in Python
## All the projects are from Udacity's Introduction to Machine Learning with TensorFlow Nanodegree Program.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### 1. Clustering

	#### 1. K Means Clustering

	In this section, the following methods/examples are implemented:

	- Identifying number of clusters (K) required for a dataset : Identifying_Clusters.ipynb
	- Effects of changing K on the result. Deciding the K by Elbow method: Changing K.ipynb
	- Feature scaling implementation in Python: Feature Scaling Example.ipynb
	- Feature Scaling using StandardScaler and MinMaxScaler: Feature Scaling.ipynb

	#### 2. Hierarchical Clustering

	In this section, the following methods/examples are implemented:

	- AgglomerativeClustering in sklearn with:
		- Ward's linkage
		- Average linkage
		- Complete linkage
	- Performance metrics: adjusted_rand_score
	- Effect of normalization on clustering
	- Dendrogram visualization with scipy
	- Visualization with Seaborn's clustermap
	
	#### 3. DBSCAN: Density based clustering
	
	In this section, the following methods/examples are implemented:
	
	- Implementat DBSCAN in sklearn.
	- Changing the values of 'epsilon' and 'min_samples'.
	- Heuristics for experimenting with DBSCAN's parameters
	
	#### 4. Gaussian mixture clustering and Cluster validation.
	
	In this section, the following methods/examples are implemented:
	
	- Implementation of Gaussian mixture clustering in sklearn
	- Comparing clustering methods using external metric 'adjected rand index'
