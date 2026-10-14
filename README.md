# Curso práctico de Machine Learning

Seis implementaciones educativas en NumPy: regresión logística, SVM lineal, Random Forest, K-Means, mezclas gaussianas y DBSCAN.

## Preparación
```bash
python -m pip install -e ".[dev]"
python generate_data.py
python supervisado/train_all.py
python no_supervisado/train_all.py
python -m pytest -q
```

Los experimentos utilizan los datasets sintéticos `well`, `semi` y `heavy`. Las métricas se exportan a `results/metrics/` y las figuras a `results/<modelo>/`.

- Clasificación: accuracy, precision, recall, F1, especificidad, balanced accuracy, ROC-AUC y average precision.
- Clustering: silhouette, Davies–Bouldin, Calinski–Harabasz, ARI y NMI cuando se conocen etiquetas.
- DBSCAN no admite predicción inductiva nativa; se evalúa con `fit` sobre el conjunto completo.
- Los algoritmos se implementan desde cero; scikit-learn se usa para particiones y evaluación.

Consulta [ML_report.md](ML_report.md) para fundamentos, ecuaciones e interpretación de resultados.

**Nota:** Las gráficas y archivos CSV/JSON deben generarse ejecutando los scripts; no se incluyen métricas ficticias.
