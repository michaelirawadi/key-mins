import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tolerance=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def fit(self, data):
        n_samples = data.shape[0]
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = data[random_indices]

        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(data)
            new_centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.k)])
            if np.all(np.abs(self.centroids - new_centroids) < self.tolerance):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, data):
        return self._assign_clusters(data)


def rmse(data, labels, centroids):
    mse = 0
    n_samples = data.shape[0]
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        mse += np.sum((cluster_points - centroids[i]) ** 2)
    mse /= n_samples
    return np.sqrt(mse)