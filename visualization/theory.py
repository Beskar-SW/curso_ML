"""Generate conceptual diagrams for the Markdown manual."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/"docs"/"images"
def save(name):
    OUT.mkdir(parents=True,exist_ok=True)
    plt.tight_layout();plt.savefig(OUT/name,dpi=170);plt.close()
def main():
    z=np.linspace(-8,8,500);p=np.exp(-np.logaddexp(0,-z))
    plt.plot(z,p);plt.axhline(.5,ls="--",alpha=.5);plt.axvline(0,ls="--",alpha=.5)
    plt.xlabel("z = w·x + b");plt.ylabel("σ(z)");plt.title("Función sigmoide");save("sigmoid.png")
    x=np.linspace(0,2,300)
    plt.plot(x,np.maximum(0,1-x),label="hinge loss (t=+1)")
    plt.plot(x,np.logaddexp(0,-x),label="logistic loss (t=+1)")
    plt.xlabel("Margen t·f(x)");plt.ylabel("Pérdida");plt.legend();save("losses.png")
    n=np.linspace(20,500,200)
    train=.97-.11*(1-np.exp(-n/90))
    val=.52+.30*(1-np.exp(-n/140))
    plt.plot(n,train,label="Entrenamiento (ilustrativo)")
    plt.plot(n,val,label="Validación (ilustrativo)")
    plt.xlabel("Tamaño de entrenamiento");plt.ylabel("Accuracy");plt.ylim(.4,1);plt.legend()
    plt.title("Curvas de aprendizaje: ejemplo conceptual, no experimental");save("learning_curves_concept.png")
if __name__=="__main__":main()
