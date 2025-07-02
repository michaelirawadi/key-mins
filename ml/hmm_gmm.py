import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)

        # Inisialisasi
        self.means = X[np.random.choice(n_samples, self.k, replace=False)]
        self.covariances = np.array([np.cov(X.T) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k  # pi
        self.resp = np.zeros((n_samples, self.k))  # gamma

        log_likelihood_old = 0

        for _ in range(self.max_iters):
            # E-step: hitung responsibilities
            for i in range(self.k):
                self.resp[:, i] = self.weights[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i])
            total_resp = np.sum(self.resp, axis=1, keepdims=True)
            self.resp /= total_resp

            # M-step: update parameter
            Nk = np.sum(self.resp, axis=0)
            self.weights = Nk / n_samples
            self.means = np.dot(self.resp.T, X) / Nk[:, np.newaxis]
            self.covariances = np.zeros((self.k, n_features, n_features))

            for i in range(self.k):
                diff = X - self.means[i]
                self.covariances[i] = (self.resp[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]

            # Cek konvergensi
            log_likelihood = np.sum(np.log(np.sum([
                self.weights[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i])
                for i in range(self.k)
            ], axis=0)))

            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            probs[:, i] = self.weights[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covariances[i])
        return np.argmax(probs, axis=1)
