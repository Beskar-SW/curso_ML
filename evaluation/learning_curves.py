"""Train-size learning curves and hyperparameter validation curves."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def _save(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    plt.tight_layout();plt.savefig(path,dpi=160);plt.close()
def learning_curve_plot(factory,X_train,y_train,X_val,y_val,path,fractions=(.2,.4,.6,.8,1.)):
    rng=np.random.default_rng(42);idx=rng.permutation(len(X_train))
    a=[];b=[];counts=[]
    for fraction in fractions:
        n=max(10,int(len(idx)*fraction));n=min(n,len(idx))
        model=factory().fit(X_train[idx[:n]],y_train[idx[:n]])
        counts.append(n);a.append(np.mean(model.predict(X_train[idx[:n]])==y_train[idx[:n]]))
        b.append(np.mean(model.predict(X_val)==y_val))
    plt.plot(counts,a,"o-",label="Entrenamiento")
    plt.plot(counts,b,"o-",label="Validación")
    plt.xlabel("Número de ejemplos");plt.ylabel("Accuracy");plt.ylim(0,1);plt.legend()
    _save(path)
def validation_curve_plot(factory,parameter,values,X_train,y_train,X_val,y_val,path):
    a=[];b=[]
    for value in values:
        model=factory(**{parameter:value}).fit(X_train,y_train)
        a.append(np.mean(model.predict(X_train)==y_train))
        b.append(np.mean(model.predict(X_val)==y_val))
    plt.plot(values,a,"o-",label="Entrenamiento")
    plt.plot(values,b,"o-",label="Validación")
    plt.xlabel(parameter);plt.ylabel("Accuracy");plt.ylim(0,1);plt.legend()
    _save(path)
