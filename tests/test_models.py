import numpy as np
import pytest
from sklearn.datasets import make_blobs
from prebuilt_models.supervised import sigmoid,LogisticRegression,LinearSVM,RandomForest
from prebuilt_models.unsupervised import KMeans,GaussianMixture,DBSCAN
def sample():
    return make_blobs(n_samples=120,centers=[(-3,-3),(3,3)],cluster_std=.5,random_state=12)
def test_sigmoid_extremes():
    p=sigmoid(np.array([-1000.,0.,1000.]))
    assert np.isfinite(p).all() and p[0]<1e-10 and p[1]==.5 and p[2]>.999
@pytest.mark.parametrize("model",[LogisticRegression(),LinearSVM(),RandomForest()])
def test_supervised(model):
    X,y=sample();model.fit(X,y)
    assert (model.predict(X)==y).mean()>.95
@pytest.mark.parametrize("model",[KMeans(),GaussianMixture()])
def test_unsupervised(model):
    X,_=sample();model.fit(X)
    assert len(model.predict(X[:7]))==7
    assert len(np.unique(model.labels_))==2
def test_dbscan():
    X,_=sample();m=DBSCAN(eps=1.,min_samples=4).fit(X)
    assert len(np.unique(m.labels_[m.labels_>=0]))==2
    with pytest.raises(NotImplementedError):m.predict(X[:3])
def test_reproducible():
    X,y=sample()
    a=RandomForest(random_state=4).fit(X,y)
    b=RandomForest(random_state=4).fit(X,y)
    np.testing.assert_array_equal(a.predict(X),b.predict(X))
