"""Evaluate clustering on the full dataset: DBSCAN is transductive."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from prebuilt_models.unsupervised import KMeans,GaussianMixture,DBSCAN
from evaluation.metrics import clustering
from evaluation.plots import clusters,save
import matplotlib.pyplot as plt
def main():
    rows=[]
    for ds in ("well","semi","heavy"):
        d=np.load(ROOT/"data"/f"dataset_{ds}.npz");X,y=d["X"],d["y"]
        for name,model in (("kmeans",KMeans()),("gmm",GaussianMixture()),("dbscan",DBSCAN(eps=.8))):
            model.fit(X);labels=model.labels_
            m=clustering(X,labels,y)
            if name=="kmeans":m["inertia"]=model.inertia_
            if name=="gmm":m["log_likelihood"]=model.score(X)
            row={"dataset":ds,"model":name,**m};rows.append(row)
            dest=ROOT/"results"/name;dest.mkdir(parents=True,exist_ok=True)
            clusters(X,labels,str(dest/f"{ds}_clusters.png"),f"{name}: {ds}",getattr(model,"centroids",None))
            if name=="kmeans":
                plt.plot(model.inertia_history_);plt.xlabel("Iteración");plt.ylabel("Inercia")
                save(str(dest/f"{ds}_inertia.png"))
            if name=="gmm":
                plt.plot(model.log_likelihood_history_);plt.xlabel("Iteración");plt.ylabel("Log-verosimilitud")
                save(str(dest/f"{ds}_likelihood.png"))
            print(row)
    out=ROOT/"results"/"metrics";out.mkdir(parents=True,exist_ok=True)
    (out/"unsupervised.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/"unsupervised.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
