"""Supervised experiments: real metrics, matrices and generalization plots."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.model_selection import train_test_split
from prebuilt_models.supervised import LogisticRegression,LinearSVM,RandomForest
from evaluation.metrics import classification
from evaluation.plots import confusion,curves,decision,history
from evaluation.learning_curves import learning_curve_plot,validation_curve_plot
def main():
    rows=[]
    for ds in ("well","semi","heavy"):
        data=np.load(ROOT/"data"/f"dataset_{ds}.npz");X,y=data["X"],data["y"]
        X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
        X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.25,stratify=y_dev,random_state=42)
        for name,factory in (("logistic",LogisticRegression),("svm",LinearSVM),("rf",RandomForest)):
            model=factory()
            if name=="rf":model.fit(X_train,y_train)
            else:model.fit(X_train,y_train,X_val,y_val)
            pred=model.predict(X_test)
            score=model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X_test)
            metrics=classification(y_test,pred,score)
            row={"dataset":ds,"model":name,**metrics,
                 "train_accuracy":float(np.mean(model.predict(X_train)==y_train)),
                 "validation_accuracy":float(np.mean(model.predict(X_val)==y_val))}
            rows.append(row)
            dest=ROOT/"results"/name;dest.mkdir(parents=True,exist_ok=True)
            base=str(dest/ds)
            confusion(y_test,pred,base+"_confusion.png")
            curves(y_test,score,base)
            decision(model,X_test,y_test,base+"_decision.png",f"{name}: {ds}")
            history(model,base+"_loss.png")
            learning_curve_plot(factory,X_train,y_train,X_val,y_val,base+"_learning_curve.png")
            param,values={"logistic":("l2",[0.,.01,.1,1.]),"svm":("lambda_param",[.001,.01,.1,1.]),"rf":("max_depth",[1,2,4,8])}[name]
            validation_curve_plot(factory,param,values,X_train,y_train,X_val,y_val,base+"_validation_curve.png")
            print(row,flush=True)
    out=ROOT/"results"/"metrics";out.mkdir(parents=True,exist_ok=True)
    (out/"supervised.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    with (out/"supervised.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
