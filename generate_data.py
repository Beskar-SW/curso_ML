import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_dataset(separation: str, n_samples=500, random_state=None):
    """Generate a 2D classification dataset.
    separation: 'well', 'semi', 'heavy' (meaning very overlapped)
    Returns X (n_samples, 2), y (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    # Means for two classes
    if separation == 'well':
        distance = 6.0
    elif separation == 'semi':
        distance = 2.5
    else:  # heavy overlap
        distance = 1.0
    mean1 = np.array([0, 0])
    mean2 = np.array([distance, distance])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    # cov = np.array([[1.0, 0], [0, 1.0]])
    # cov = np.array([[1.0, -0.5], [-0.5, 1.0]])
    X1 = rng.multivariate_normal(mean1, cov, n_samples // 2)
    X2 = rng.multivariate_normal(mean2, cov, n_samples // 2)
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    # shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def save_dataset(X, y, name):
    os.makedirs('data', exist_ok=True)
    np.savez(os.path.join('data', f'{name}.npz'), X=X, y=y)

if __name__ == "__main__":
    for level in ['well', 'semi', 'heavy']:
        X, y = generate_dataset(level, random_state=42)
        save_dataset(X, y, f'dataset_{level}')
        # Plot and save
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='deep', legend=False)
        plt.title(f'Dataset ({level} separation)')
        plt.savefig(os.path.join('data', f'dataset_{level}.png'))
        plt.close()

