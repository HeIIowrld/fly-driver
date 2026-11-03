import copy
import pytest
from flybrain_drive.diagnostic_sim_clock_pairs import summarize_clock_pairs

def sample():
    first={'source_clock_before':1.,'source_clock_after':1.001,'sim_time_ms':10000.,'car_timestamp_ms':9995.}
    last={'source_clock_before':1.020,'source_clock_after':1.021,'sim_time_ms':10000.,'car_timestamp_ms':9995.}
    return {'training_admissible':False,'actor_stream_created':False,'views':[{'view_id':v,'render_started_clock':1.002,'render_call_finished_clock':1.019,'diagnostic_clock_pair_before':copy.deepcopy(first),'diagnostic_clock_pair_after':copy.deepcopy(last)} for v in ('front','left','right','rear')]}

def test_reports_candidate_difference_without_admission():
    r=summarize_clock_pairs(sample());assert len(r['views'])==4 and r['views'][0]['before_car_minus_sim_ms']==-5 and r['physical_exposure_sync_verified'] is False and r['training_admissible'] is False

def test_rejects_missing_pair_and_bad_order_and_actor_data():
    x=sample();x['views'][0]['diagnostic_clock_pair_before']=None
    with pytest.raises(ValueError):summarize_clock_pairs(x)
    x=sample();x['views'][0]['render_call_finished_clock']=1.03
    with pytest.raises(ValueError):summarize_clock_pairs(x)
    x=sample();x['training_admissible']=True
    with pytest.raises(ValueError):summarize_clock_pairs(x)

def test_clock_reversal_reported_without_false_sync():
    x=sample();x['views'][0]['diagnostic_clock_pair_after']['sim_time_ms']=9999.
    r=summarize_clock_pairs(x);assert r['views'][0]['sim_clock_reversed_during_render'] and not r['physical_exposure_sync_verified']
