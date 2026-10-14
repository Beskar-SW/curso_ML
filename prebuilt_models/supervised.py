import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, tol=1e-4):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.w) + self.b
            y_pred = sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            w_old, b_old = self.w.copy(), self.b
            self.w -= self.lr * dw
            self.b -= self.lr * db
            if np.linalg.norm(self.w - w_old) < self.tol and abs(self.b - b_old) < self.tol:
                break
        return self

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_pred = sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

# Simple linear SVM using the primal form and sub‑gradient descent
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to {-1, 1}
        y_ = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.epochs):
            for idx, xi in enumerate(X):
                condition = y_[idx] * (np.dot(xi, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(xi, y_[idx]))
                    self.b -= self.lr * y_[idx]
        return self

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return (approx >= 0).astype(int)

# Very lightweight Random Forest using bagging of decision stumps (single‑feature thresholds)
class RandomForest:
    def __init__(self, n_estimators=25, max_depth=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.features = []
        self.thresholds = []
        self.preds = []

    def _train_stump(self, X, y, rng):
        # Choose a random feature
        feature_idx = rng.randint(0, X.shape[1])
        # Choose a threshold midway between min and max of that feature
        col = X[:, feature_idx]
        thresh = (col.min() + col.max()) / 2
        # Simple prediction: 1 if feature >= thresh else 0
        pred = (col >= thresh).astype(int)
        return feature_idx, thresh, pred

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            f_idx, thresh, pred = self._train_stump(X_boot, y_boot, rng)
            self.features.append(f_idx)
            self.thresholds.append(thresh)
            self.preds.append(pred)
        return self

    def predict(self, X):
        votes = np.zeros(X.shape[0], dtype=int)
        for f_idx, thresh in zip(self.features, self.thresholds):
            votes += (X[:, f_idx] >= thresh).astype(int)
        # Majority vote (>= half predicts class 1)
        return (votes >= (self.n_estimators / 2)).astype(int)

# Exportable helpers
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = LinearSVM()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForest()
    model.fit(X_train, y_train)
    return model

