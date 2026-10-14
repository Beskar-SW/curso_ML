"""Render actual measured metrics and linked figures into an experiment report."""
from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/"results"/"metrics"/name).open(encoding="utf-8",newline="") as fp:
        return list(csv.DictReader(fp))
def table(rows,cols):
    header="| "+" | ".join(cols)+" |"
    rule="| "+" | ".join("---" for _ in cols)+" |"
    return "\n".join([header,rule]+["| "+" | ".join(str(r.get(k,"—")) for k in cols)+" |" for r in rows])
def main():
    sup=read("supervised.csv");uns=read("unsupervised.csv")
    lines=["# Resultados experimentales del curso","",
        "Datos medidos al ejecutar los scripts. Los conjuntos de prueba se reservan para evaluación final.",
        "## 1. Clasificación","",
        table(sup,["dataset","model","train_accuracy","validation_accuracy","accuracy","precision","recall","f1","roc_auc","average_precision"])]
    for row in sup:
        name=row["model"];ds=row["dataset"];base=f"{name}/{ds}"
        lines.extend(["",f"### {name.upper()} — {ds}","",
            f"Accuracy entrenamiento: {float(row['train_accuracy']):.3f}; validación: {float(row['validation_accuracy']):.3f}; prueba: {float(row['accuracy']):.3f}.",
            "Una diferencia marcada entre entrenamiento y validación puede indicar sobreajuste; valores bajos en ambos sugieren subajuste. Verifica también la curva de aprendizaje.",
            f"![Matriz de confusión]({base}_confusion.png)",
            f"![Curva de aprendizaje]({base}_learning_curve.png)",
            f"![Curva de validación]({base}_validation_curve.png)",
            f"![ROC]({base}_roc.png)",
            f"![Precision-Recall]({base}_pr.png)",
            f"![Frontera de decisión]({base}_decision.png)"])
        if name in ("logistic","svm"):
            lines.append(f"![Pérdida por época]({base}_loss.png)")
    lines.extend(["","## 2. Clustering","",
        "Las métricas internas no se calculan si solo existe un clúster válido. En DBSCAN se excluye el ruido para métricas internas y se registra por separado.",
        table(uns,["dataset","model","n_clusters","noise_fraction","silhouette","davies_bouldin","calinski_harabasz","ari","nmi"])])
    for row in uns:
        name=row["model"];ds=row["dataset"];base=f"{name}/{ds}"
        lines.extend(["",f"### {name.upper()} — {ds}","",f"![Clústeres y ruido]({base}_clusters.png)"])
        if name=="kmeans":lines.append(f"![Convergencia de inercia]({base}_inertia.png)")
        if name=="gmm":lines.append(f"![Convergencia de log-verosimilitud]({base}_likelihood.png)")
    lines.extend(["","## 3. Guía de interpretación",
        "Comparar métricas entre datasets, inspeccionar clases minoritarias, ruido y sensibilidad a hiperparámetros. No confundir grupos encontrados con etiquetas verdaderas.",
        "Consultar [manual teórico](../ML_report.md) para derivaciones y limitaciones."])
    path=ROOT/"results"/"EXPERIMENT_REPORT.md"
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Reporte generado: {path}")
if __name__=="__main__":main()
