# Explicación de **generate_data.py**

Este script genera tres conjuntos de datos sintéticos en 2‑D que difieren en el grado de separación entre las dos clases. A continuación se describen los conceptos clave y por qué se usan determinadas expresiones.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
```

### `generate_dataset(separation, n_samples=500, random_state=None)`
- **`separation`**: cadena que indica el nivel de separación de las clases.
  - `'well'` → clases muy separadas.
  - `'semi'` → semi‑solapadas.
  - `'heavy'` → mucho solapamiento.
- **`n_samples`**: número total de puntos (se divide a la mitad para cada clase).
- **`random_state`**: semilla para reproducibilidad.

#### 1. Elección de la distancia entre medias
```python
if separation == 'well':
    distance = 5.0
elif separation == 'semi':
    distance = 2.5
else:  # heavy overlap
    distance = 1.0
```
- `distance` es la distancia euclídea entre los vectores de media **`mean1`** y **`mean2`**.  Un valor mayor genera clases más separadas.

#### 2. Medias y covarianza
```python
mean1 = np.array([0, 0])
mean2 = np.array([distance, distance])
cov = np.array([[1.0, 0.5], [0.5, 1.0]])
```
- `mean1` y `mean2` son **arrays de 2 valores** porque los datos son 2‑dimensionales (x, y).
- `cov` es la **matriz de covarianza** que controla la forma y la orientación de los clusters. Los valores `0.5` en los off‑diagonal (valores fuera de la diagonal principal) indican correlación entre las dimensiones.

#### 3. Generación de los puntos
```python
X1 = rng.multivariate_normal(mean1, cov, n_samples // 2)
X2 = rng.multivariate_normal(mean2, cov, n_samples // 2)
```
- `rng.multivariate_normal(mean, cov, size)` genera **muestras** que siguen una distribución normal multivariada con la media y covarianza especificadas.
- Se pasa `n_samples // 2` porque **la mitad de los puntos pertenece a cada clase**. Así obtenemos un balance 0/1.

#### 4. Construcción del dataset completo
```python
X = np.vstack([X1, X2])
 y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
```
- `np.vstack` **apila verticalmente** los arrays de características de ambas clases para crear la matriz `X` de forma `(n_samples, 2)`.
- `np.hstack` crea el vector de etiquetas `y` con `0`s para la primera clase y `1`s para la segunda.

#### 5. Mezclar los datos
```python
idx = rng.permutation(n_samples)
X = X[idx]
 y = y[idx]
```
- `rng.permutation` devuelve una permutación aleatoria de índices, lo que **mezcla** (shuffle) los ejemplos para que no estén ordenados por clase.

### Guardado y visualización
```python
os.makedirs('data', exist_ok=True)
np.savez(os.path.join('data', f'{name}.npz'), X=X, y=y)
```
- Los datos se guardan en formato comprimido `.npz` para fácil carga posterior.
- Además se genera un **scatter plot** con seaborn y se guarda como PNG para inspección rápida.

---

Con esta explicación el lector puede entender cómo y por qué se crean los tres conjuntos de datos (`well`, `semi`, `heavy`) que luego se usan para entrenar y comparar los modelos supervisados y no supervisados.

