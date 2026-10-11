# K-Means Clustering

## Introduction

K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters.

This project implements K-Means clustering using the Iris dataset.

## Dataset Selection

The Iris dataset was selected from the three datasets available in the repository because it contains four numerical flower measurements and three known flower species.

The numerical features make it suitable for demonstrating distance-based clustering.

## Objectives

- Understand unsupervised learning.
- Implement K-Means clustering.
- Understand cluster centers.
- Apply feature scaling.
- Use the Elbow Method.
- Evaluate clusters using the Silhouette Score.
- Visualize clusters using PCA.
- Compare clusters with known species labels after training.

## How K-Means Works

K-Means follows these steps:

1. Select the number of clusters, K.
2. Initialize K cluster centers.
3. Assign each data point to its nearest center.
4. Calculate new centers using the mean of the assigned points.
5. Repeat the assignment and update steps until the centers stabilize or the iteration limit is reached.

## Feature Scaling

K-Means uses distances to determine which observations belong together.

StandardScaler standardizes the numerical features to approximately zero mean and unit variance.

Scaling prevents features with larger numerical ranges from dominating the distance calculations.

## Choosing the Number of Clusters

### Elbow Method

The Elbow Method uses inertia to help choose K.

Inertia measures the sum of squared distances between observations and their assigned cluster centers.

As K increases, inertia generally decreases. The elbow indicates a point where adding more clusters provides smaller improvements.

### Silhouette Score

The Silhouette Score measures how well an observation fits within its assigned cluster compared with other clusters.

Its values range from -1 to 1. A higher score generally indicates better-separated clusters.

Neither method guarantees a perfect value of K, so the results should be interpreted together.

## Model Training

The notebook uses:

`KMeans(n_clusters=3, random_state=42, n_init=10)`

Three clusters are used for this demonstration because the Iris dataset has three known species.

The target labels are not used to train the model.

## Cluster Centers

K-Means calculates a center for each cluster.

The notebook displays the centers in standardized units and converts them back to the original feature units for easier interpretation.

## Comparing Clusters with Species

After training, the notebook compares cluster assignments with the known species labels using a cross-tabulation.

This is only a descriptive comparison. Cluster IDs are arbitrary and do not automatically correspond to the target labels.

## Libraries Used

- pandas
- NumPy
- Matplotlib
- scikit-learn
