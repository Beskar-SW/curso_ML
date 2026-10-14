# Machine Learning

## 1. Introducción a Machine Learning (ML)
Machine Learning es una disciplina que permite a las computadoras **aprender patrones** a partir de datos sin ser programadas explícitamente para cada tarea. Se clasifica principalmente en:
- **Aprendizaje supervisado** – el algoritmo recibe ejemplos con etiquetas (p.ej., clase 0 o 1).
- **Aprendizaje no supervisado** – el algoritmo solo observa los datos y busca estructura (p.ej., grupos).

---

## 2. Modelos supervisados usados en este proyecto
En este repositorio utilizamos tres algoritmos clásicos que no requieren librerías externas más allá de **numpy**:

### 2.1 Regresión Logística
- **Qué hace:** Modela la probabilidad de pertenecer a la clase 1 mediante una función sigmoide.
- **Parámetros clave:** `lr` (tasa de aprendizaje), `epochs` (número máximo de iteraciones), `tol` (tolerancia para convergencia).
- **Código relevante:**
```python
model = LogisticRegression(lr=0.1, epochs=2000, tol=1e-4)
model.fit(X_train, y_train)
``` 

### 2.2 Support Vector Machine (SVM) lineal
- **Qué hace:** Encuentra un hiperplano que maximiza el margen entre clases.
- **Parámetros clave:** `lr` (tasa de aprendizaje), `lambda_param` (regularización), `epochs`.
- **Código relevante:**
```python
model = LinearSVM(lr=0.001, lambda_param=0.01, epochs=1000)
model.fit(X_train, y_train)
```

### 2.3 Bosque Aleatorio (Random Forest) simplificado
- **Qué hace:** Construye varios "stumps" (árboles de profundidad 1) y combina sus votos.
- **Parámetros clave:** `n_estimators` (número de árboles), `max_depth` (aquí siempre 1), `random_state`.
- **Código relevante:**
```python
model = RandomForest(n_estimators=25, max_depth=1, random_state=42)
model.fit(X_train, y_train)
```

---

## 3. Parámetros que el usuario puede modificar
| Modelo | Parámetro | Descripción | Lugar en el código |
|--------|-----------|-------------|--------------------|
| Regresión Logística | `lr` | Paso de actualización de pesos. Valores pequeños = entrenamiento más lento pero estable. | `LogisticRegression.__init__`
| | `epochs` | Máximo número de iteraciones. | idem |
| SVM | `lr` | Tasa de aprendizaje para el descenso de sub‑gradiente. | `LinearSVM.__init__`
| | `lambda_param` | Coeficiente de regularización (penaliza pesos grandes). | idem |
| Random Forest | `n_estimators` | Cuántos árboles (stumps) crear. Más árboles → mejor generalización, mayor tiempo. | `RandomForest.__init__`
| | `random_state` | Semilla para reproducibilidad. | idem |

---

## 4. Cómo se ven estos parámetros en el código
Todos los modelos están definidos en **`prebuilt_models/supervised.py`**. Cada clase recibe sus hiper‑parámetros en el constructor y los usa dentro de los bucles `for _ in range(self.epochs)` (para LR y SVM) o al crear los árboles (para Random Forest).

---

## 5. Conclusiones
Este proyecto muestra, de forma mínima y sin dependencias externas, cómo entrenar y visualizar tres algoritmos supervisados sobre conjuntos de datos sintéticos con diferentes grados de superposición. Cambiando los hiper‑parámetros descritos se puede observar el efecto en la exactitud y la forma de los límites de decisión.

---

*Este documento está pensado para ser compilado a PDF usando LaTeX o pandoc.*

