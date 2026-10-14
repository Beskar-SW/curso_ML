import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Simple K-Means implementation
class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        # Init centroids randomly from points
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]
        for _ in range(self.max_iter):
            # Assign clusters
            distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                                      for i in range(self.n_clusters)])
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        self.labels_ = labels
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

# Simple Gaussian Mixture Model (EM) with diagonal covariances
class GaussianMixture:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means = None
        self.covs = None
        self.weights = None

    def _gaussian(self, X, mean, cov):
        # diagonal covariance
        var = np.diag(cov)
        coeff = 1.0 / np.sqrt((2 * np.pi) ** X.shape[1] * np.prod(var))
        exp_term = np.exp(-0.5 * np.sum(((X - mean) ** 2) / var, axis=1))
        return coeff * exp_term

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        # Initialize parameters
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        self.covs = np.array([np.cov(X.T) for _ in range(self.n_components)])
        self.weights = np.full(self.n_components, 1 / self.n_components)
        log_likelihood_old = -np.inf
        for _ in range(self.max_iter):
            # E-step
            resp = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                resp[:, k] = self.weights[k] * self._gaussian(X, self.means[k], self.covs[k])
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp = resp / resp_sum
            # M-step
            Nk = resp.sum(axis=0)
            self.weights = Nk / n_samples
            self.means = (resp.T @ X) / Nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covs[k] = np.diag((resp[:, k][:, None] * diff ** 2).sum(axis=0) / Nk[k])
            # Compute log‑likelihood
            ll = np.log(resp_sum).sum()
            if abs(ll - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = ll
        self.labels_ = np.argmax(resp, axis=1)
        return self

    def predict(self, X):
        return self.labels_

# Simple DBSCAN implementation (Euclidean distance)
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)  # -1 = noise
        cluster_id = 0
        # Precompute distance matrix
        dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        for i in range(n_samples):
            if labels[i] != -1:
                continue
            neighbors = np.where(dist_matrix[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue
            # Expand cluster
            labels[i] = cluster_id
            seeds = set(neighbors)
            seeds.discard(i)
            while seeds:
                current = seeds.pop()
                if labels[current] == -1:
                    labels[current] = cluster_id
                if labels[current] != -1:
                    continue
                labels[current] = cluster_id
                current_neighbors = np.where(dist_matrix[current] <= self.eps)[0]
                if len(current_neighbors) >= self.min_samples:
                    seeds.update(current_neighbors)
            cluster_id += 1
        self.labels_ = labels
        return self

    def predict(self, X):
        return self.labels_

# Export helper functions
def train_kmeans(X):
    model = KMeans(random_state=42)
    model.fit(X)
    return model

def train_gmm(X):
    model = GaussianMixture(random_state=42)
    model.fit(X)
    return model

def train_dbscan(X):
    model = DBSCAN(eps=0.8, min_samples=5)
    model.fit(X)
    return model

