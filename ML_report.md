# Curso completo de Machine Learning: teoría, implementación y evaluación

**Nivel:** introductorio a intermedio · **Lenguaje:** Python y NumPy · **Algoritmos:** seis · **Modalidad:** lectura + experimentos reproducibles.

Este documento no sustituye los experimentos: primero lee la teoría, luego ejecuta los comandos del [README](README.md) y abre [el informe experimental](results/EXPERIMENT_REPORT.md) que se genera con métricas e imágenes reales. Los diagramas SVG incluidos aquí son **conceptuales**, no resultados inventados.

## Índice

1. ¿Qué es aprender a partir de datos?
2. Matemáticas, optimización y particiones
3. Regresión logística y función sigmoide
4. SVM lineal
5. Árboles y Random Forest
6. K-Means
7. Gaussian Mixture Models
8. DBSCAN
9. Evaluación supervisada
10. Evaluación no supervisada
11. Overfitting, underfitting y curvas de aprendizaje
12. Prácticas, interpretación y limitaciones

---

## 1. ¿Qué es aprender a partir de datos?

En aprendizaje automático se ajusta una función $f_\\theta$ con parámetros $\\theta$ a partir de observaciones. El objetivo no es memorizar los datos, sino **generalizar** a observaciones nuevas.

![Flujo de aprendizaje supervisado](docs/images/supervised_workflow.svg)

**Aprendizaje supervisado:** se observan pares $(x_i,y_i)$. En clasificación binaria, $y_i\\in\\{0,1\\}$; la tarea es predecir una etiqueta a partir de características. En este curso, las coordenadas $(x_1,x_2)$ son las características y el grupo generador es la etiqueta.

![Flujo de aprendizaje no supervisado](docs/images/unsupervised_workflow.svg)

**Aprendizaje no supervisado:** solo se observan $x_i$. Se busca estructura, densidad o agrupaciones. Los grupos obtenidos no son automáticamente categorías semánticas. En el laboratorio las etiquetas originales se conservan **solo para evaluar externamente** el clustering mediante ARI y NMI; nunca se entregan al ajuste no supervisado.

### Tres datasets

- **well:** dos gaussianas relativamente separadas; la clasificación debería resultar más sencilla.
- **semi:** separación intermedia, con superposición parcial.
- **heavy:** gran superposición; existe error irreducible porque observaciones similares pueden pertenecer a clases diferentes.

Se generan mediante `generate_data.py` con semilla 42, 500 observaciones y covarianza compartida. Consulta las imágenes `data/dataset_well.png`, `data/dataset_semi.png` y `data/dataset_heavy.png`.

## 2. Matemáticas, optimización y particiones

Un modelo ajusta parámetros minimizando una **función de pérdida** $L(\\theta)$, que penaliza predicciones erróneas. En descenso del gradiente:

$$\\theta_{t+1}=\\theta_t-\\eta\\nabla_\\theta L(\\theta_t)$$

Aquí $\\eta$ es la tasa de aprendizaje. Una tasa demasiado grande puede oscilar; una muy pequeña puede tardar mucho. **Regularización** significa penalizar complejidad, por ejemplo $\\lambda\\|w\\|_2^2/2$.

Los **parámetros** se aprenden (pesos, centroides, covarianzas); los **hiperparámetros** se eligen (tasa, profundidad, $k$, epsilon). Para estimar generalización, separar entrenamiento, validación y prueba. La selección de hiperparámetros debe usar validación; el conjunto de prueba queda reservado para evaluación final.

En clasificación, la partición es estratificada. Las curvas de aprendizaje reentrenan modelos sobre fracciones de entrenamiento y miden tanto entrenamiento como validación.

## 3. Regresión logística

**Pregunta:** ¿cuál es la probabilidad de que una observación pertenezca a la clase 1?

El modelo combina linealmente características:

$$z=w_1x_1+w_2x_2+b.$$

La **función sigmoide** convierte $z$ en un valor entre 0 y 1:

$$\\sigma(z)=\\frac{1}{1+e^{-z}}.$$

![Sigmoide conceptual](docs/images/sigmoid.svg)

Si $z=0$, la salida es $0.5$; si $z=\\ln(3)$, es $0.75$. Un valor de 0.75 significa probabilidad estimada del 75% **según el modelo**, no certeza de que el ejemplo sea positivo. El umbral habitual de decisión es 0.5, aunque puede modificarse para priorizar recall o precision.

La derivada de la sigmoide es:

$$\\sigma'(z)=\\sigma(z)(1-\\sigma(z)).$$

La **entropía cruzada binaria** penaliza probabilidades incorrectas:

$$L=-\\frac1n\\sum_i\\left[y_i\\log p_i+(1-y_i)\\log(1-p_i)\\right].$$

Con $p_i=\\sigma(w^\\top x_i+b)$, los gradientes son:

$$\\frac{\\partial L}{\\partial w}=\\frac1nX^\\top(p-y),\\qquad
\\frac{\\partial L}{\\partial b}=\\frac1n\\sum_i(p_i-y_i).$$

La implementación de `prebuilt_models/supervised.py` utiliza NumPy y una forma numéricamente estable basada en `logaddexp` para evitar problemas con logaritmos de cero.

**Ventajas:** interpretación de coeficientes, probabilidades, entrenamiento eficiente. **Limitaciones:** frontera lineal en las características originales; requiere ingeniería de variables para relaciones no lineales. **Hiperparámetros:** `lr`, `epochs`, `l2`, `tol`.

**Experimento:** observa `results/logistic/well_loss.png` y `well_learning_curve.png`. Una pérdida de validación que empeora mientras la de entrenamiento disminuye es señal de posible sobreajuste. Comprueba también la matriz de confusión y ROC.

## 4. Support Vector Machine (SVM) lineal

**Pregunta:** ¿qué hiperplano separa mejor las clases y deja un margen amplio?

La frontera se expresa como $w^\\top x+b=0$. Se usan etiquetas transformadas $t_i=2y_i-1$, con valores -1 y +1. El margen funcional es $t_i(w^\\top x_i+b)$; una predicción correcta con margen al menos 1 no incurre en hinge loss.

$$\\ell_i=\\max(0,1-t_i(w^\\top x_i+b)).$$

La función objetivo regularizada es:

$$J(w,b)=\\frac{\\lambda}{2}\\|w\\|_2^2+\\frac1n\\sum_i\\ell_i.$$

**Interpretación:** el primer término limita pesos grandes; el segundo penaliza violaciones del margen. El parámetro $\\lambda$ modifica el equilibrio. Los ejemplos cercanos al margen o mal clasificados determinan el gradiente de la pérdida hinge.

**Diferencia frente a logística:** la SVM entrega una puntuación de margen, no una probabilidad calibrada. Para ROC/PR se usa `decision_function`. No debe interpretarse un margen de 2 como 200% de probabilidad.

**Ventajas:** buena separación lineal y control de complejidad. **Limitaciones:** no resuelve fronteras no lineales sin kernels o transformaciones; la versión educativa es lineal. **Hiperparámetros:** `lambda_param`, `lr`, `epochs`.

**Experimento:** compara `results/svm/*_decision.png` con `*_validation_curve.png` al variar regularización.

## 5. Árboles de decisión y Random Forest

Un árbol pregunta sucesivamente si una característica supera un umbral. Cada división separa el conjunto en dos subconjuntos. En clasificación binaria, la impureza de Gini es:

$$G(S)=1-p_0^2-p_1^2=2p_1(1-p_1).$$

Si todos los ejemplos de un nodo son de una clase, $G=0$. Si las dos clases están al 50%, $G=0.5$. El árbol selecciona la división con menor impureza ponderada de los hijos.

**Random Forest** crea varios árboles con muestras bootstrap (muestreo con reemplazo) y subconjuntos aleatorios de características. La predicción se obtiene promediando las probabilidades estimadas en las hojas. Esta aleatoriedad reduce la correlación entre árboles.

**Ejemplo:** si tres árboles estiman $0.2$, $0.8$ y $0.9$ para la clase positiva, el promedio es $0.633$ y el umbral 0.5 produce clase 1.

**Ventajas:** fronteras no lineales y menor necesidad de escalado. **Limitaciones:** puede sobreajustarse con árboles profundos y pocos datos; es menos interpretable que un modelo lineal. **Hiperparámetros:** `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.

**Diagnóstico de sobreajuste:** un bosque no tiene una pérdida por épocas comparable a descenso del gradiente. Usa `results/rf/*_learning_curve.png` y `*_validation_curve.png`, donde la complejidad se modifica mediante profundidad.

## 6. K-Means (no supervisado)

**Pregunta:** ¿cómo agrupar observaciones alrededor de $k$ centros?

K-Means minimiza la suma de distancias cuadráticas al centro asignado, denominada **inercia**:

$$J=\\sum_{i=1}^n\\min_{1\\leq j\\leq k}\\|x_i-\\mu_j\\|_2^2.$$

El procedimiento alterna dos operaciones: (1) asignar cada punto al centro más cercano; (2) recalcular cada centro como media de los puntos asignados. Repetir hasta convergencia o alcanzar el máximo de iteraciones.

![K-Means: asignación y actualización](docs/images/kmeans.svg)

**Ejemplo:** para los puntos 1, 2, 8 y 9, con $k=2$ y centros iniciales 1 y 9, las asignaciones naturales son {1,2} y {8,9}; los centros actualizados son 1.5 y 8.5.

**Ventajas:** simple y eficiente. **Limitaciones:** requiere $k$; sensible a inicialización, escala y valores atípicos; favorece grupos compactos. **Hiperparámetros:** `n_clusters`, `max_iter`, `tol`.

**Diagnósticos:** `results/kmeans/*_clusters.png` y `*_inertia.png`. La curva de inercia por iteración muestra convergencia; el **método del codo** requiere comparar distintos valores de $k$, no iteraciones. Silhouette, ARI y NMI ayudan a evaluar estructura y correspondencia con etiquetas conocidas.

## 7. Gaussian Mixture Model (GMM)

**Pregunta:** ¿puede cada punto pertenecer a distintos grupos con probabilidades diferentes?

Un GMM modela la distribución como suma ponderada de gaussianas:

$$p(x)=\\sum_{k=1}^K\\pi_k\\,\\mathcal N(x\\mid\\mu_k,\\Sigma_k),\\quad\\sum_k\\pi_k=1.$$

Los pesos $\\pi_k$ son proporciones de mezcla; $\\mu_k$ son medias; $\\Sigma_k$ describe dispersión. La **responsabilidad** $r_{ik}$ es la probabilidad posterior de que el componente $k$ explique la observación $i$:

$$r_{ik}=\\frac{\\pi_k\\mathcal N(x_i\\mid\\mu_k,\\Sigma_k)}{\\sum_j\\pi_j\\mathcal N(x_i\\mid\\mu_j,\\Sigma_j)}.$$

El algoritmo **Expectation-Maximization (EM)** alterna dos fases:

1. **E-step:** calcular responsabilidades con parámetros actuales.
2. **M-step:** actualizar pesos, medias y covarianzas usando responsabilidades.

![GMM: pertenencia probabilística](docs/images/gmm.svg)

**Ejemplo:** una observación cerca de la frontera puede recibir responsabilidades 0.55 y 0.45, mientras K-Means la asignaría exclusivamente a uno de los grupos.

**Ventajas:** agrupamiento suave y modelado probabilístico. **Limitaciones:** puede converger a óptimos locales, requiere fijar número de componentes y puede sufrir singularidades numéricas. **Esta implementación usa covarianzas diagonales**, no gaussianas con orientación arbitraria. **Hiperparámetros:** `n_components`, `max_iter`, `reg_covar`, `tol`.

**Diagnósticos:** `results/gmm/*_clusters.png` y `*_likelihood.png`. La log-verosimilitud debe interpretarse con cuidado: más componentes pueden mejorar ajuste sin mejorar generalización. BIC/AIC serían extensiones razonables.

## 8. DBSCAN (no supervisado)

**Pregunta:** ¿qué regiones tienen suficiente densidad y cuáles son ruido?

DBSCAN utiliza `eps` (radio de vecindad) y `min_samples` (número mínimo de puntos vecinos, incluido el propio punto). Distingue:

- **Núcleo:** tiene al menos `min_samples` observaciones en su vecindad.
- **Frontera:** no es núcleo, pero está dentro del radio de un núcleo.
- **Ruido:** no pertenece a ningún clúster; etiqueta -1.

Los grupos se construyen expandiendo regiones de puntos núcleo conectados por densidad. Los puntos frontera pueden asignarse a un grupo aunque no expandan vecinos.

![DBSCAN: núcleo, frontera y ruido](docs/images/dbscan.svg)

**Ejemplo:** en una nube densa de 20 puntos, `min_samples=5` y `eps` suficientemente grande generan puntos núcleo. Un punto aislado lejos de todos ellos se etiqueta como ruido.

**Ventajas:** no requiere indicar $k$; detecta formas irregulares y ruido. **Limitaciones:** sensible a `eps`, `min_samples` y escalas; difícil con densidades variables. **Importante:** DBSCAN no tiene una operación `predict(X_test)` estándar. En el laboratorio se aplica `fit` al conjunto que se desea agrupar y se evalúa la partición obtenida, sin fingir predicción inductiva.

**Diagnósticos:** `results/dbscan/*_clusters.png`, proporción de ruido, número de grupos, silhouette sobre puntos no ruidosos y ARI/NMI si existen etiquetas de referencia. No se usa matriz de confusión de clasificación para DBSCAN.

## 9. Métricas de clasificación

![Interpretación de TP, TN, FP y FN](docs/images/confusion_matrix.svg)

| Métrica | Fórmula | ¿Qué responde? |
|---|---|---|
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ | ¿Qué proporción total acertó? |
| Precision | $TP/(TP+FP)$ | ¿Cuántos positivos predichos eran correctos? |
| Recall / sensibilidad | $TP/(TP+FN)$ | ¿Cuántos positivos reales detectó? |
| Especificidad | $TN/(TN+FP)$ | ¿Cuántos negativos reales detectó? |
| F1 | $2PR/(P+R)$ | ¿Qué equilibrio existe entre precision y recall? |
| Balanced accuracy | $(Recall+Especificidad)/2$ | ¿Cómo rinde entre clases? |

**Ejemplo numérico:** con TP=40, TN=45, FP=5 y FN=10, accuracy=0.85, precision≈0.889, recall=0.80, especificidad=0.90 y F1≈0.842.

**ROC:** dibuja TPR frente a FPR para varios umbrales; ROC-AUC mide capacidad de ordenación. **Precision–Recall:** representa el compromiso entre precision y recall; suele informar más cuando la clase positiva es escasa. **Average precision** resume la curva PR. Un buen AUC no demuestra probabilidades calibradas.

Para cada modelo y dataset se generan matrices de confusión, ROC, PR y fronteras de decisión, además de CSV y JSON de métricas.

## 10. Métricas de clustering

| Métrica | ¿Qué mide? | Interpretación |
|---|---|---|
| Silhouette | Cohesión frente a separación | Mayor suele ser mejor, intervalo [-1,1] |
| Davies–Bouldin | Dispersión relativa entre grupos | Menor suele ser mejor |
| Calinski–Harabasz | Separación intergrupo frente a intragrupo | Mayor suele ser mejor |
| ARI | Coincidencia con clases de referencia corregida por azar | 1 significa acuerdo perfecto |
| NMI | Información compartida entre clústeres y clases | 1 significa acuerdo perfecto |
| Inercia | Distancia cuadrática a centroides de K-Means | Menor para el mismo $k$ y dataset |
| Log-verosimilitud | Ajuste probabilístico de GMM | Mayor para mismo conjunto y modelo comparable |
| Proporción de ruido | Fracción etiquetada -1 en DBSCAN | Depende del objetivo; no tiene óptimo universal |

Las métricas internas como silhouette no se definen con un solo grupo. Para DBSCAN, se excluye ruido de las métricas internas y se informa explícitamente su proporción. ARI/NMI usan etiquetas conocidas solo en evaluación.

## 11. Overfitting, underfitting y diagnóstico por modelo

**Underfitting:** entrenamiento y validación malos; puede faltar capacidad, variables o entrenamiento. **Overfitting:** entrenamiento muy bueno pero validación sustancialmente peor; puede existir exceso de complejidad, datos insuficientes o problemas de distribución.

![Curvas conceptuales de generalización](docs/images/generalization.svg)

**Curva de aprendizaje:** eje X = cantidad de ejemplos de entrenamiento; eje Y = métrica. Dos curvas, entrenamiento y validación. Si convergen a un nivel bajo, investiga subajuste. Si persiste una brecha grande, investiga sobreajuste. Estas son heurísticas, no diagnósticos automáticos.

**Curva de validación:** eje X = hiperparámetro; eje Y = rendimiento en entrenamiento y validación. En este curso se explora `l2` (logística), `lambda_param` (SVM) y `max_depth` (Random Forest).

**Pérdida por época:** se usa en logística y SVM porque optimizan iterativamente. No se inventa una curva por épocas para Random Forest.

**¿Y los no supervisados?** No existe una noción universal de error de clasificación de entrenamiento/validación sin etiquetas. Sus diagnósticos adecuados son convergencia, sensibilidad a $k$, número de componentes, `eps`, estabilidad del clustering y, para GMM, verosimilitud en datos retenidos. No etiquetes una curva de inercia como curva de overfitting.

## 12. Prácticas guiadas

**Práctica A: clasificación.** Ejecuta `python supervisado/train_all.py`. Abre las nueve matrices de confusión y compara cómo aumentan FP/FN al pasar de `well` a `heavy`. Después observa ROC y PR: ¿los rankings siguen siendo útiles aunque disminuya accuracy?

**Práctica B: generalización.** Abre `results/rf/heavy_learning_curve.png` y `results/rf/heavy_validation_curve.png`. ¿Qué profundidad produce el mayor rendimiento de validación? ¿Existe una brecha importante con entrenamiento? Repite para logística y SVM.

**Práctica C: clustering.** Ejecuta `python no_supervisado/train_all.py`. Compara K-Means, GMM y DBSCAN en los tres datasets. Interpreta silhouette y ARI por separado: un agrupamiento geométricamente compacto puede no coincidir con las clases generadoras.

**Práctica D: DBSCAN.** Cambia `eps` de 0.3 a 0.8 y 1.5. Registra número de grupos, ruido y silhouette. Explica por qué un solo clúster puede invalidar silhouette.

**Práctica E: GMM.** Examina las responsabilidades y la evolución de log-verosimilitud. Explica por qué pertenencia suave no es lo mismo que una clase real.

### Comandos en orden

Con pip: `python -m pip install -e ".[dev]"`; con uv: `uv venv` y `uv pip install -e ".[dev]"`. Después:

```bash
python generate_data.py
python visualization/theory.py
python -m pytest -q
python supervisado/train_all.py
python no_supervisado/train_all.py
python course_report.py
```

También puedes usar `python run_course.py` después de instalar. **Abre `results/EXPERIMENT_REPORT.md` para ver las imágenes reales y las tablas generadas**.

### Límites y buenas prácticas

Los ejemplos son sintéticos y bidimensionales. Las implementaciones priorizan transparencia matemática sobre eficiencia. Random Forest puede resultar lento porque evalúa umbrales; DBSCAN calcula distancias por pares y consume memoria cuadrática. Para problemas reales: escalado ajustado solo con entrenamiento, validación cruzada, análisis de desbalance, comparación con bibliotecas optimizadas y evaluación de estabilidad. Los resultados de este curso deben obtenerse ejecutando el código, no suponerse.
