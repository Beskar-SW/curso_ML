# Machine Learning Demo

This repository demonstrates supervised and unsupervised learning models on three synthetic datasets with varying class separation:

- **well**: well‑separated classes
- **semi**: partially overlapping classes
- **heavy**: heavily overlapping classes

## Folder Structure

```
curso_ML/
├─ prebuilt_models/          # Simple model implementations using only numpy
│   ├─ __init__.py
│   ├─ supervised.py        # Logistic Regression, Linear SVM, Random Forest
│   └─ unsupervised.py      # K‑Means, Gaussian Mixture, DBSCAN
├─ data/                     # Generated datasets (created by generate_data.py)
├─ results/                  # Plots produced by training scripts
├─ generate_data.py          # Creates synthetic datasets & example plots
├─ supervisado/              # Supervised learning models
│   ├─ logistic_regression/
│   │   ├─ __init__.py
│   │   └─ train.py
│   ├─ svm/
│   │   ├─ __init__.py
│   │   └─ train.py
│   ├─ random_forest/
│   │   ├─ __init__.py
│   │   └─ train.py
│   └─ train_all.py         # Runs all supervised models on all datasets
└─ no_supervisado/           # Unsupervised clustering models
    ├─ kmeans/
    │   ├─ __init__.py
    │   └─ train.py (optional)
    ├─ gaussian_mixture/
    │   ├─ __init__.py
    │   └─ train.py (optional)
    ├─ dbscan/
    │   ├─ __init__.py
    │   └─ train.py (optional)
    └─ train_all.py         # Runs all unsupervised models on all datasets
```

## Quick Start

1. **Generate the data**
   ```powershell
   python generate_data.py
   ```
   This creates three `.npz` files in the `data/` folder and example scatter plots.

2. **Run supervised models**
   ```powershell
   python supervisado\train_all.py
   ```
   Individual models are also available, e.g. `python supervisado\logistic_regression\train.py`.

3. **Run unsupervised models**
   ```powershell
   python no_supervisado\train_all.py
   ```

All scripts save result plots under `results/<model_name>/`.

## Dependencies

Only the following libraries are used:
- `numpy`
- `matplotlib`
- `seaborn`

Feel free to explore or modify the model implementations in `prebuilt_models/`.

