import numpy as np
import pytest
from flybrain_drive.synthetic_domain_interventions import MODES,intervene

@pytest.mark.parametrize('mode',MODES)
def test_intervention_preserves_input_and_three_other_views(mode):
    views=[np.full((180,320,3),100,dtype=np.uint8) for _ in range(4)]
    out=intervene(views,mode)
    assert all(np.all(x==100) for x in views)
    assert all(np.array_equal(out[i],views[i]) for i in (1,2,3))
    assert all(np.array_equal(a,b) for a,b in zip(out,intervene(views,mode)))
    assert np.array_equal(out[0],views[0]) == (mode=='baseline')

def test_reject_invalid_input():
    with pytest.raises(ValueError):intervene([], 'baseline')
    with pytest.raises(ValueError):intervene([np.zeros((180,320,3))]*4,'baseline')

def test_accepts_generator_array():
    views=np.full((4,180,320,3),100,dtype=np.uint8)
    out=intervene(views,'front_cockpit')
    assert len(out)==4 and np.array_equal(out[1],views[1]) and np.all(views==100)
