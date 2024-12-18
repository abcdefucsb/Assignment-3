import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn.metrics import silhouette_score 
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    K_list=np.arange(1, 11)
    inertia = []
    silhouette_score_list=[]
    for k in K_list:
        model = KMeans(k, "euclidean", max_iter=300, tol=1e-4)
        model.fit(og_iris)
        inertia.append(model.get_error())
        if k>1:
            silhouette_score_list.append(silhouette_score(og_iris,model.predict(og_iris)))
    print("silhouette score list \n")
    print(silhouette_score_list)
    print("\n")    
    plt.plot(K_list, inertia, marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
    # Fit model
    

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    K_list=np.arange(1, 11)
    inertia = []
    for k in K_list:
        model = KMeans(k, "euclidean", max_iter=300, tol=1e-4)
        model.fit(df)
        inertia.append(model.get_error())
        if k>1:
            plot_3d_clusters(df,model.predict(df),"kmeans",silhouette_score(df, model.predict(df)))

    plt.plot(K_list, inertia, marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
    # You can choose which scoring method you'd like to use here: Silhouette (explained below)
    
    
    # Plot your data using plot_3d_clusters in visualization.py

    
    # Try different numbers of clusters

    
    # Plot the elbow plot

    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 
    3
    
    Reasoning: 
    3 is the elbow point. Reduction in inertia slows after k=3
    Based on the Elbow Method and Silhouette Scores, we are attempting to find the optimal number of clusters (K) that minimizes inertia and maximizes the silhouette score.
    The Elbow Method will show us where the inertia begins to level off, indicating the point where adding more clusters does not improve the fit significantly.

    Silhouette score explanation:
    The Silhouette Score is a measure of how well-separated the clusters are in a K-means clustering model. It evaluates two key factors:
    Cohesion: How close the points in a cluster are to each other.
    Separation: How distinct a cluster is from other clusters. 
    A higher silhouette score (close to +1) indicates well-separated and cohesive clusters, while a score close to 0 suggests overlapping clusters, and negative scores indicate poor clustering.
    It provides an intuitive and quantitative way to assess clustering quality without needing ground truth labels.
    It helps select the optimal number of clusters by comparing scores across different values of K.
    
    
    
    """

    
if __name__ == "__main__":
    main()