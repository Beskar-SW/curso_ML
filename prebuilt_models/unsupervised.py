"""Educational clustering estimators. DBSCAN intentionally has no out-of-sample predict."""
import numpy as np
from scipy.special import logsumexp

class KMeans:
    def __init__(self,n_clusters=2,max_iter=100,tol=1e-4,random_state=42):
        self.n_clusters,self.max_iter,self.tol,self.random_state=n_clusters,max_iter,tol,random_state
    def fit(self,X):
        X=np.asarray(X,float); rng=np.random.default_rng(self.random_state)
        if self.n_clusters>len(X): raise ValueError("n_clusters exceeds sample count")
        self.centroids=X[rng.choice(len(X),self.n_clusters,replace=False)].copy()
        self.inertia_history_=[]
        for _ in range(self.max_iter):
            labels=self.predict(X)
            centers=np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else X[rng.integers(len(X))] for i in range(self.n_clusters)])
            shift=np.linalg.norm(centers-self.centroids)
            self.centroids=centers
            self.inertia_history_.append(float(np.sum((X-self.centroids[self.predict(X)])**2)))
            if shift<self.tol: break
        self.labels_=self.predict(X)
        self.inertia_=float(np.sum((X-self.centroids[self.labels_])**2))
        return self
    def predict(self,X):
        X=np.asarray(X,float)
        return np.argmin(np.sum((X[:,None,:]-self.centroids[None,:,:])**2,axis=2),axis=1)

class GaussianMixture:
    def __init__(self,n_components=2,max_iter=100,tol=1e-4,random_state=42,reg_covar=1e-6):
        self.n_components,self.max_iter,self.tol,self.random_state,self.reg_covar=n_components,max_iter,tol,random_state,reg_covar
    def _log_components(self,X):
        d=X.shape[1]; out=[]
        for k in range(self.n_components):
            var=np.maximum(self.covs[k],self.reg_covar)
            out.append(np.log(max(self.weights[k],1e-300))-.5*(d*np.log(2*np.pi)+np.log(var).sum()+(((X-self.means[k])**2)/var).sum(axis=1)))
        return np.column_stack(out)
    def fit(self,X):
        X=np.asarray(X,float); n,d=X.shape
        rng=np.random.default_rng(self.random_state)
        if self.n_components>n: raise ValueError("Too many components")
        self.means=X[rng.choice(n,self.n_components,replace=False)].copy()
        self.covs=np.tile(np.var(X,axis=0)+self.reg_covar,(self.n_components,1))
        self.weights=np.full(self.n_components,1/self.n_components)
        self.log_likelihood_history_=[]
        for _ in range(self.max_iter):
            logp=self._log_components(X)
            lognorm=logsumexp(logp,axis=1)
            ll=float(lognorm.sum())
            self.log_likelihood_history_.append(ll)
            resp=np.exp(logp-lognorm[:,None])
            nk=np.maximum(resp.sum(axis=0),1e-12)
            self.weights=nk/n
            self.means=(resp.T@X)/nk[:,None]
            for k in range(self.n_components):
                self.covs[k]=(resp[:,k,None]*(X-self.means[k])**2).sum(axis=0)/nk[k]+self.reg_covar
            if len(self.log_likelihood_history_)>1 and abs(ll-self.log_likelihood_history_[-2])<self.tol: break
        self.labels_=self.predict(X)
        self.lower_bound_=self.score(X)
        return self
    def predict_proba(self,X):
        logp=self._log_components(np.asarray(X,float))
        return np.exp(logp-logsumexp(logp,axis=1)[:,None])
    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)
    def score(self,X):
        return float(logsumexp(self._log_components(np.asarray(X,float)),axis=1).mean())

class DBSCAN:
    def __init__(self,eps=0.5,min_samples=5):
        self.eps,self.min_samples=eps,min_samples
    def fit(self,X):
        X=np.asarray(X,float); n=len(X)
        distances=np.linalg.norm(X[:,None,:]-X[None,:,:],axis=2)
        neighbors=[np.flatnonzero(distances[i]<=self.eps) for i in range(n)]
        core=np.array([len(nb)>=self.min_samples for nb in neighbors])
        labels=np.full(n,-1,dtype=int); visited=np.zeros(n,bool); cid=0
        from collections import deque
        for i in range(n):
            if visited[i]: continue
            visited[i]=True
            if not core[i]: continue
            labels[i]=cid
            queue=deque(neighbors[i]); queued=set(neighbors[i])
            while queue:
                j=queue.popleft()
                if not visited[j]:
                    visited[j]=True
                    if core[j]:
                        for k in neighbors[j]:
                            if k not in queued:
                                queue.append(k); queued.add(k)
                if labels[j]==-1: labels[j]=cid
            cid+=1
        self.labels_=labels; self.core_sample_indices_=np.flatnonzero(core); self.components_=X[core].copy()
        return self
    def fit_predict(self,X):
        return self.fit(X).labels_
    def predict(self,X):
        raise NotImplementedError("DBSCAN has no native out-of-sample predict; use fit_predict on the evaluated dataset.")

def train_kmeans(X): return KMeans().fit(X)
def train_gmm(X): return GaussianMixture().fit(X)
def train_dbscan(X): return DBSCAN(eps=0.8,min_samples=5).fit(X)
