import sys, os
# Add project root to PYTHONPATH so that prebuilt_models can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from prebuilt_models.supervised import train_logistic_regression, train_svm, train_random_forest

def load_dataset(name):
    data = np.load(os.path.join('data', f'dataset_{name}.npz'))
    return data['X'], data['y']

def main(model_type, dataset_name):
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if model_type == 'logistic':
        model = train_logistic_regression(X_train, y_train)
    elif model_type == 'svm':
        model = train_svm(X_train, y_train)
    elif model_type == 'rf':
        model = train_random_forest(X_train, y_train)
    else:
        raise ValueError('Unknown model')
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f'Accuracy: {acc:.4f}')
    # Plot test points colored by true label and overlay predictions
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=y_test, palette='deep', style=y_pred, markers=True, legend='full')
    plt.title(f'{model_type.upper()} on {dataset_name} separation (Acc {acc:.2f})')
    out_dir = os.path.join('results', model_type)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_plot.png'))
    plt.close()

if __name__ == '__main__':
    for ds in ['well', 'semi', 'heavy']:
        for mt in ['logistic', 'svm', 'rf']:
            main(mt, ds)

