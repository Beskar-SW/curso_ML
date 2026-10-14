# Curso completo de Machine Learning (NumPy)

Este repositorio permite **estudiar, entrenar, evaluar y comparar seis algoritmos** implementados desde cero: regresión logística, SVM lineal, Random Forest, K-Means, Gaussian Mixture Model (GMM) y DBSCAN.

**Empieza por [ML_report.md](ML_report.md)**: teoría, ecuaciones, figuras y guía para interpretar cada experimento. Los resultados numéricos y gráficos se generan ejecutando los experimentos, no se inventan.

## Requisitos

Python 3.10 o superior. Ejecuta los comandos desde la raíz del repositorio. Si vienes de otra rama: `git switch feature/curso-ml-completo`.

### Opción A: uv (recomendada)

```bash
uv venv
uv pip install -e ".[dev]"
uv run --no-sync python generate_data.py
uv run --no-sync python visualization/theory.py
uv run --no-sync python -m pytest -q
uv run --no-sync python supervisado/train_all.py
uv run --no-sync python no_supervisado/train_all.py
uv run --no-sync python course_report.py
```

**Por qué `--no-sync`:** se utiliza la instalación editable del paso anterior sin intentar sincronizar un `uv.lock` antiguo. Para un entorno completamente bloqueado, actualiza el lock con `uv lock` y luego usa `uv sync --extra dev` antes de `uv run`.

### Opción B: pip

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python generate_data.py
python visualization/theory.py
python -m pytest -q
python supervisado/train_all.py
python no_supervisado/train_all.py
python course_report.py
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python generate_data.py
python visualization/theory.py
python -m pytest -q
python supervisado/train_all.py
python no_supervisado/train_all.py
python course_report.py
```

### Ejecución en un solo comando (después de instalar)

```bash
python run_course.py
```

El orden de ejecución es: **(1)** generar datos; **(2)** diagramas de teoría; **(3)** pruebas; **(4)** modelos supervisados; **(5)** modelos no supervisados; **(6)** informe experimental con imágenes y tablas. Si una etapa falla, el ejecutor se detiene.

## ¿Qué archivos debo abrir después?

| Resultado | Ruta |
|---|---|
| Curso teórico ilustrado | `ML_report.md` |
| Informe experimental generado | `results/EXPERIMENT_REPORT.md` |
| Métricas supervisadas (por modelo y dataset) | `results/metrics/supervised.csv` |
| Métricas no supervisadas | `results/metrics/unsupervised.csv` |
| Matrices de confusión (9) | `results/{logistic,svm,rf}/*_confusion.png` |
| ROC y precision-recall (18) | `results/{logistic,svm,rf}/*_roc.png`, `*_pr.png` |
| Curvas de aprendizaje (9) | `results/{logistic,svm,rf}/*_learning_curve.png` |
| Curvas de validación / complejidad (9) | `results/{logistic,svm,rf}/*_validation_curve.png` |
| Curvas de pérdida (logística y SVM) | `results/{logistic,svm}/*_loss.png` |
| Agrupamientos no supervisados (9) | `results/{kmeans,gmm,dbscan}/*_clusters.png` |
| Convergencia K-Means y GMM | `results/kmeans/*_inertia.png`, `results/gmm/*_likelihood.png` |

Cada patrón representa tres datasets (`well`, `semi`, `heavy`). **Las curvas de aprendizaje comparan entrenamiento y validación**: no se debe diagnosticar sobreajuste mirando solo el resultado de test. **Para clustering no se define accuracy de entrenamiento ni una matriz de confusión de clasificación**: se emplean silhouette, ARI, NMI, ruido y estabilidad, según el algoritmo.

## Orden de lectura y prácticas

1. Secciones 1–3 de `ML_report.md`: conceptos, sigmoide y regresión logística.
2. Secciones 4–5: SVM, árboles y Random Forest.
3. Secciones 6–8: K-Means, GMM y DBSCAN.
4. Secciones 9–11: métricas, generalización y comparación experimental.
5. Abrir `results/EXPERIMENT_REPORT.md`, comparar gráficas y resolver los ejercicios.

**Reproducibilidad:** semillas fijas en la generación y partición. Los datasets existentes pueden regenerarse. Las gráficas se generan localmente; las imágenes conceptuales SVG del manual ya están versionadas.

**Advertencia:** este es un laboratorio educativo; no se ha certificado para producción. Las implementaciones deliberadamente simples pueden ser lentas en datasets grandes.
