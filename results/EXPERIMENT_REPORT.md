# Resultados experimentales del curso

Datos medidos al ejecutar los scripts. Los conjuntos de prueba se reservan para evaluación final.
## 1. Clasificación

| dataset | model | train_accuracy | validation_accuracy | accuracy | precision | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| well | logistic | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| well | svm | 0.9966666666666667 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| well | rf | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| semi | logistic | 0.9266666666666666 | 0.95 | 0.93 | 0.9387755102040817 | 0.92 | 0.9292929292929293 | 0.9836 | 0.9851749685846367 |
| semi | svm | 0.93 | 0.93 | 0.94 | 0.9074074074074074 | 0.98 | 0.9423076923076923 | 0.9852 | 0.9867246821336604 |
| semi | rf | 0.9733333333333334 | 0.93 | 0.93 | 0.9574468085106383 | 0.9 | 0.9278350515463918 | 0.983 | 0.9852285933611096 |
| heavy | logistic | 0.7333333333333333 | 0.72 | 0.72 | 0.7391304347826086 | 0.68 | 0.7083333333333334 | 0.7979999999999999 | 0.8012925293031228 |
| heavy | svm | 0.7466666666666667 | 0.69 | 0.71 | 0.6909090909090909 | 0.76 | 0.7238095238095238 | 0.8032 | 0.8054381164836648 |
| heavy | rf | 0.83 | 0.68 | 0.71 | 0.7560975609756098 | 0.62 | 0.6813186813186813 | 0.7735999999999998 | 0.7506951189853439 |

### LOGISTIC — well

Accuracy entrenamiento: 1.000; validación: 1.000; prueba: 1.000.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](logistic/well_confusion.png)
![Curva de aprendizaje](logistic/well_learning_curve.png)
![Curva de validación](logistic/well_validation_curve.png)
![ROC](logistic/well_roc.png)
![Precision-Recall](logistic/well_pr.png)
![Frontera de decisión](logistic/well_decision.png)
![Pérdida por época](logistic/well_loss.png)

### SVM — well

Accuracy entrenamiento: 0.997; validación: 1.000; prueba: 1.000.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](svm/well_confusion.png)
![Curva de aprendizaje](svm/well_learning_curve.png)
![Curva de validación](svm/well_validation_curve.png)
![ROC](svm/well_roc.png)
![Precision-Recall](svm/well_pr.png)
![Frontera de decisión](svm/well_decision.png)
![Pérdida por época](svm/well_loss.png)

### RF — well

Accuracy entrenamiento: 1.000; validación: 1.000; prueba: 1.000.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](rf/well_confusion.png)
![Curva de aprendizaje](rf/well_learning_curve.png)
![Curva de validación](rf/well_validation_curve.png)
![ROC](rf/well_roc.png)
![Precision-Recall](rf/well_pr.png)
![Frontera de decisión](rf/well_decision.png)

### LOGISTIC — semi

Accuracy entrenamiento: 0.927; validación: 0.950; prueba: 0.930.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](logistic/semi_confusion.png)
![Curva de aprendizaje](logistic/semi_learning_curve.png)
![Curva de validación](logistic/semi_validation_curve.png)
![ROC](logistic/semi_roc.png)
![Precision-Recall](logistic/semi_pr.png)
![Frontera de decisión](logistic/semi_decision.png)
![Pérdida por época](logistic/semi_loss.png)

### SVM — semi

Accuracy entrenamiento: 0.930; validación: 0.930; prueba: 0.940.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](svm/semi_confusion.png)
![Curva de aprendizaje](svm/semi_learning_curve.png)
![Curva de validación](svm/semi_validation_curve.png)
![ROC](svm/semi_roc.png)
![Precision-Recall](svm/semi_pr.png)
![Frontera de decisión](svm/semi_decision.png)
![Pérdida por época](svm/semi_loss.png)

### RF — semi

Accuracy entrenamiento: 0.973; validación: 0.930; prueba: 0.930.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](rf/semi_confusion.png)
![Curva de aprendizaje](rf/semi_learning_curve.png)
![Curva de validación](rf/semi_validation_curve.png)
![ROC](rf/semi_roc.png)
![Precision-Recall](rf/semi_pr.png)
![Frontera de decisión](rf/semi_decision.png)

### LOGISTIC — heavy

Accuracy entrenamiento: 0.733; validación: 0.720; prueba: 0.720.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](logistic/heavy_confusion.png)
![Curva de aprendizaje](logistic/heavy_learning_curve.png)
![Curva de validación](logistic/heavy_validation_curve.png)
![ROC](logistic/heavy_roc.png)
![Precision-Recall](logistic/heavy_pr.png)
![Frontera de decisión](logistic/heavy_decision.png)
![Pérdida por época](logistic/heavy_loss.png)

### SVM — heavy

Accuracy entrenamiento: 0.747; validación: 0.690; prueba: 0.710.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](svm/heavy_confusion.png)
![Curva de aprendizaje](svm/heavy_learning_curve.png)
![Curva de validación](svm/heavy_validation_curve.png)
![ROC](svm/heavy_roc.png)
![Precision-Recall](svm/heavy_pr.png)
![Frontera de decisión](svm/heavy_decision.png)
![Pérdida por época](svm/heavy_loss.png)

### RF — heavy

Accuracy entrenamiento: 0.830; validación: 0.680; prueba: 0.710.
Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.
![Matriz de confusión](rf/heavy_confusion.png)
![Curva de aprendizaje](rf/heavy_learning_curve.png)
![Curva de validación](rf/heavy_validation_curve.png)
![ROC](rf/heavy_roc.png)
![Precision-Recall](rf/heavy_pr.png)
![Frontera de decisión](rf/heavy_decision.png)

## 2. Clustering

Las métricas internas no se calculan si solo existe un clúster válido. En DBSCAN se excluye el ruido para métricas internas y se registra por separado.
| dataset | model | n_clusters | noise_fraction | silhouette | davies_bouldin | calinski_harabasz | ari | nmi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| well | kmeans | 2 | 0.0 | 0.7962845207008684 | 0.2814019059603452 | 4738.544374440517 | 1.0 | 1.0 |
| well | gmm | 2 | 0.0 | 0.7962845207008684 | 0.2814019059603452 | 4738.544374440517 | 1.0 | 1.0 |
| well | dbscan | 2 | 0.012 | 0.8019008522132991 | 0.27546170888752386 | 4987.969533107196 | 0.9761430645009139 | 0.9501359342408232 |
| semi | kmeans | 2 | 0.0 | 0.5582050719992985 | 0.6139977364754874 | 1032.3511495121043 | 0.7459869884675356 | 0.6417355325770399 |
| semi | gmm | 2 | 0.0 | 0.5582314883131585 | 0.6140069614298391 | 1032.191671843719 | 0.7390771168167994 | 0.6341127948846277 |
| semi | dbscan | 1 | 0.008 |  |  |  | -6.361927761592913e-05 | 0.0 |
| heavy | kmeans | 2 | 0.0 | 0.42390808159767995 | 0.8688737262279917 | 503.19951111530315 | 0.1955672691219472 | 0.15373946136588063 |
| heavy | gmm | 2 | 0.0 | 0.42365541860493794 | 0.8628542155182853 | 493.35533726871523 | 0.21753745287760967 | 0.17912543396495537 |
| heavy | dbscan | 1 | 0.008 |  |  |  | -6.361927761592913e-05 | 0.0 |

### KMEANS — well

![Clústeres y ruido](kmeans/well_clusters.png)
![Convergencia de inercia](kmeans/well_inertia.png)

### GMM — well

![Clústeres y ruido](gmm/well_clusters.png)
![Convergencia de log-verosimilitud](gmm/well_likelihood.png)

### DBSCAN — well

![Clústeres y ruido](dbscan/well_clusters.png)

### KMEANS — semi

![Clústeres y ruido](kmeans/semi_clusters.png)
![Convergencia de inercia](kmeans/semi_inertia.png)

### GMM — semi

![Clústeres y ruido](gmm/semi_clusters.png)
![Convergencia de log-verosimilitud](gmm/semi_likelihood.png)

### DBSCAN — semi

![Clústeres y ruido](dbscan/semi_clusters.png)

### KMEANS — heavy

![Clústeres y ruido](kmeans/heavy_clusters.png)
![Convergencia de inercia](kmeans/heavy_inertia.png)

### GMM — heavy

![Clústeres y ruido](gmm/heavy_clusters.png)
![Convergencia de log-verosimilitud](gmm/heavy_likelihood.png)

### DBSCAN — heavy

![Clústeres y ruido](dbscan/heavy_clusters.png)

## 3. Guía de interpretación
Comparar métricas entre datasets, inspeccionar clases minoritarias, ruido y sensibilidad a hiperparámetros. No confundir grupos encontrados con etiquetas verdaderas.
Consultar [manual teórico](../ML_report.md) para derivaciones y limitaciones.
