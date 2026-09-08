import numpy as np
import pandas as pd
from liftlab import covariate_hash, split_groups, transformed_outcome, curve, s_features

def test_duplicates_stay_together():
    frame=pd.DataFrame({'f0':[1.,2.,1.], 'f1':[3.,4.,3.]})
    hashes=covariate_hash(frame)
    assert hashes[0]==hashes[2]
    assert split_groups(hashes)[0]==split_groups(hashes)[2]

def test_split_order_independent():
    h=np.arange(1000,dtype=np.uint64)
    assert np.array_equal(split_groups(h)[::-1], split_groups(h[::-1]))
    assert set(split_groups(h))=={'train','dev','test'}

def test_ipw_known_balanced_trial():
    assert transformed_outcome(np.array([1,0,0,0]),np.array([1,1,0,0]),.5).mean()==.5

def test_zero_effect_zero_curve():
    f,g,a,q=curve(np.array([2.,1.]),np.zeros(2))
    assert a==q==0 and np.all(g==0)

def test_treatment_interactions_only():
    x=np.ones((4,12))
    assert s_features(x,0).shape==(4,25)
    assert np.all(s_features(x,0)[:,13:]==0)

def test_bad_propensity_rejected():
    import pytest
    with pytest.raises(ValueError): transformed_outcome(np.ones(2),np.ones(2),0)
