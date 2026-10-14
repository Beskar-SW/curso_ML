import numpy as np
from evaluation.metrics import classification,clustering
def test_binary_metrics():
    m=classification([0,0,1,1],[0,1,1,1],[.1,.8,.9,.7])
    assert m["accuracy"]==.75
    assert m["recall"]==1.
    assert m["specificity"]==.5
def test_degenerate_clusters():
    x=np.arange(12).reshape(6,2)
    m=clustering(x,np.zeros(6,dtype=int))
    assert m["silhouette"] is None
