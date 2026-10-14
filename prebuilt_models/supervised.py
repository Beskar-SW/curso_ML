"""Educational NumPy classifiers with a scikit-learn-like API."""
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0, -z))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, tol=1e-6, l2=0.0):
        self.lr, self.epochs, self.tol, self.l2 = lr, epochs, tol, l2
    def fit(self, X, y, X_val=None, y_val=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.w, self.b = np.zeros(X.shape[1]), 0.
        self.history_ = {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            dw = X.T @ (p-y)/len(y) + self.l2*self.w
            db = np.mean(p-y)
            delta = self.lr*np.linalg.norm(dw)
            self.w -= self.lr*dw
            self.b -= self.lr*db
            self.history_["train_loss"].append(self.loss(X,y))
            if X_val is not None:
                self.history_["val_loss"].append(self.loss(X_val,y_val))
            if delta < self.tol and abs(self.lr*db) < self.tol:
                break
        return self
    def loss(self, X, y):
        z = np.asarray(X) @ self.w + self.b
        return float(np.mean(np.logaddexp(0,z)-np.asarray(y)*z)+self.l2*np.dot(self.w,self.w)/2)
    def predict_proba(self, X):
        p = sigmoid(np.asarray(X)@self.w+self.b)
        return np.column_stack([1-p,p])
    def decision_function(self, X):
        return np.asarray(X)@self.w+self.b
    def predict(self, X):
        return (self.predict_proba(X)[:,1]>=0.5).astype(int)

class LinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, epochs=1000, random_state=42):
        self.lr, self.lambda_param, self.epochs, self.random_state = lr, lambda_param, epochs, random_state
    def fit(self, X, y, X_val=None, y_val=None):
        X=np.asarray(X,float); t=2*np.asarray(y)-1
        self.w=np.zeros(X.shape[1]); self.b=0.
        self.history_={"train_loss":[],"val_loss":[]}
        for epoch in range(self.epochs):
            margin=t*(X@self.w+self.b)
            active=margin<1
            dw=self.lambda_param*self.w - (X[active].T@t[active])/len(X)
            db=-np.sum(t[active])/len(X)
            step=self.lr/np.sqrt(1+epoch/100)
            self.w-=step*dw; self.b-=step*db
            self.history_["train_loss"].append(self.loss(X,(t+1)/2))
            if X_val is not None: self.history_["val_loss"].append(self.loss(X_val,y_val))
        return self
    def loss(self,X,y):
        t=2*np.asarray(y)-1
        return float(self.lambda_param*np.dot(self.w,self.w)/2+np.maximum(0,1-t*self.decision_function(X)).mean())
    def decision_function(self,X):
        return np.asarray(X)@self.w+self.b
    def predict(self,X):
        return (self.decision_function(X)>=0).astype(int)

class _Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value,self.feature,self.threshold,self.left,self.right=value,feature,threshold,left,right

class RandomForest:
    def __init__(self,n_estimators=25,max_depth=5,min_samples_split=2,max_features="sqrt",random_state=42):
        self.n_estimators,self.max_depth,self.min_samples_split=n_estimators,max_depth,min_samples_split
        self.max_features,self.random_state=max_features,random_state
    def _grow(self,X,y,depth,rng):
        p=float(y.mean())
        if depth>=self.max_depth or len(y)<self.min_samples_split or p in (0.,1.):
            return _Node(value=p)
        nf=X.shape[1]
        m=max(1,int(np.sqrt(nf))) if self.max_features=="sqrt" else nf
        best=(np.inf,None,None,None)
        for j in rng.choice(nf,size=m,replace=False):
            vals=np.unique(X[:,j])
            if len(vals)<2: continue
            thresholds=(vals[:-1]+vals[1:])/2
            for threshold in thresholds:
                mask=X[:,j]<=threshold
                a,b=y[mask],y[~mask]
                if not len(a) or not len(b): continue
                g=lambda v:2*v.mean()*(1-v.mean())
                score=(len(a)*g(a)+len(b)*g(b))/len(y)
                if score<best[0]: best=(score,j,threshold,mask)
        _,j,threshold,mask=best
        if j is None: return _Node(value=p)
        return _Node(feature=j,threshold=threshold,
            left=self._grow(X[mask],y[mask],depth+1,rng),
            right=self._grow(X[~mask],y[~mask],depth+1,rng))
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,int)
        rng=np.random.default_rng(self.random_state)
        self.trees=[]
        for _ in range(self.n_estimators):
            ix=rng.integers(len(X),size=len(X))
            self.trees.append(self._grow(X[ix],y[ix],0,rng))
        return self
    def _prob(self,x,node):
        while node.value is None:
            node=node.left if x[node.feature]<=node.threshold else node.right
        return node.value
    def predict_proba(self,X):
        X=np.asarray(X,float)
        p=np.mean([[self._prob(x,tree) for tree in self.trees] for x in X],axis=1)
        return np.column_stack((1-p,p))
    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=0.5).astype(int)

def train_logistic_regression(X_train,y_train):
    return LogisticRegression().fit(X_train,y_train)
def train_svm(X_train,y_train):
    return LinearSVM().fit(X_train,y_train)
def train_random_forest(X_train,y_train):
    return RandomForest().fit(X_train,y_train)
