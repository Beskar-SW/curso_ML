import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from prebuilt_models.supervised import train_svm
from sklearn.model_selection import train_test_split

def load_dataset(name):
    data = np.load(os.path.join('data', f'dataset_{name}.npz'))
    return data['X'], data['y']

def main(dataset_name):
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_svm(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f'SVM Accuracy on {dataset_name}: {acc:.4f}')
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=y_test, palette='deep', style=y_pred, markers=True, legend='full')
    plt.title(f'SVM - {dataset_name} separation (Acc {acc:.2f})')
    out_dir = os.path.join('results', 'svm')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_plot.png'))
    plt.close()

if __name__ == '__main__':
    for ds in ['well', 'semi', 'heavy']:
        main(ds)

