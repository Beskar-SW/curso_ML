"""Non-interactive publication-ready experiment plots."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn.model_selection import learning_curve
def save(path):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()
def confusion(y,p,path):
    ConfusionMatrixDisplay.from_predictions(y,p,labels=[0,1],cmap="Blues")
    save(path)
def curves(y,s,base):
    if len(np.unique(y))<2: return
    RocCurveDisplay.from_predictions(y,s);save(base+"_roc.png")
    PrecisionRecallDisplay.from_predictions(y,s);save(base+"_pr.png")
def decision(model,X,y,path,title):
    if X.shape[1]!=2:return
    x0,x1=np.meshgrid(np.linspace(X[:,0].min()-1,X[:,0].max()+1,180),np.linspace(X[:,1].min()-1,X[:,1].max()+1,180))
    z=model.predict(np.column_stack([x0.ravel(),x1.ravel()])).reshape(x0.shape)
    plt.contourf(x0,x1,z,alpha=.2);plt.scatter(X[:,0],X[:,1],c=y,s=14,cmap="coolwarm",edgecolors="none")
    plt.title(title);save(path)
def history(model,path):
    h=getattr(model,"history_",None)
    if not h:return
    plt.plot(h["train_loss"],label="Entrenamiento")
    if h["val_loss"]:plt.plot(h["val_loss"],label="Validación")
    plt.xlabel("Época");plt.ylabel("Pérdida");plt.legend();save(path)
def clusters(X,labels,path,title,centers=None):
    plt.scatter(X[:,0],X[:,1],c=labels,cmap="tab10",s=15)
    if centers is not None:plt.scatter(centers[:,0],centers[:,1],marker="X",s=120,c="black")
    plt.title(title);save(path)
