import pytest
from flybrain_drive.training_source_guard import require_training_source

@pytest.mark.parametrize('recording,raw,labels',[
 ({'training_admissible':False},{'continuous_40ms_training_admissible':True},{'training_admissible':True}),
 ({'training_admissible':True},{'continuous_40ms_training_admissible':False},{'training_admissible':True}),
 ({'training_admissible':True},{'continuous_40ms_training_admissible':True},{'training_admissible':False}),
 ({'training_admissible':1},{'continuous_40ms_training_admissible':True},{'training_admissible':True}),
 ({},{},{}),
])
def test_unqualified_or_ambiguous_sources_are_rejected(recording,raw,labels):
    with pytest.raises(ValueError):require_training_source(recording=recording,raw_audit=raw,label_audit=labels)

def test_distinct_explicit_admissions_are_required():
    require_training_source(recording={'training_admissible':True},raw_audit={'continuous_40ms_training_admissible':True},label_audit={'training_admissible':True})
