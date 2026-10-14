import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from prebuilt_models.unsupervised import train_kmeans, train_gmm, train_dbscan
from sklearn.model_selection import train_test_split


def load_dataset(name):
    data = np.load(os.path.join('data', f'dataset_{name}.npz'))
    return data['X']

def main(model_type, dataset_name):
    X = load_dataset(dataset_name)
    # For unsupervised we still split to visualize train/test but labels are not used
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    if model_type == 'kmeans':
        model = train_kmeans(X_train)
    elif model_type == 'gmm':
        model = train_gmm(X_train)
    elif model_type == 'dbscan':
        model = train_dbscan(X_train)
    else:
        raise ValueError('Unknown model')
    # Predict clusters on test set
    labels = model.predict(X_test)
    # Plot test points colored by cluster label
    plt.figure(figsize=(6,5))
    palette = sns.color_palette('deep', np.unique(labels).size)
    sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=labels, palette=palette, legend='full')
    plt.title(f'{model_type.upper()} on {dataset_name} separation')
    out_dir = os.path.join('results', model_type)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_plot.png'))
    plt.close()

if __name__ == '__main__':
    for ds in ['well', 'semi', 'heavy']:
        for mt in ['kmeans', 'gmm', 'dbscan']:
            main(mt, ds)

