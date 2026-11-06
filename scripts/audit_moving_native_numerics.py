"""Independent original prediction and raw hidden-state numerical evidence audit."""
import argparse
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.diagnostic);r=read(root/'result.json');m=read(root/'manifest.json')['config']
    assert r['status']=='completed_moving_native_numerics_diagnostic' and sha256(root/'executed-entrypoint.py')==m['program_sha256']
    shadow=Path(m['arguments']['shadow']);sr=read(shadow/'result.json')
    assert sha256(shadow/'result.json')==m['shadow_result_sha256'] and sha256(shadow/'owner-ticker-events.json')==sr['ticker_events_sha256']
    for file,key in [('arrays.npz','arrays_sha256'),('records.json','records_sha256')]:assert sha256(root/file)==r[key]
    events=read(shadow/'owner-ticker-events.json');valid=[e for e in events if e['status']=='shadow_output'];expected=np.asarray([e['prediction'] for e in valid])
    assert len(events)==len(valid)+1 and events[-1]['status']=='rejected'
    errors={}
    with np.load(root/'arrays.npz') as arrays:
        reference=arrays['prediction_flush_False'];candidate=arrays['prediction_flush_True']
        assert reference.shape==expected.shape and candidate.shape==(len(events),3)
        for mode,value in [('reference',reference),('candidate',candidate[:len(valid)])]:
            assert np.isfinite(value).all();delta=float(abs(value-expected).max());assert delta<=2e-6;errors[mode]=delta
        byte_equal=reference.tobytes()==candidate[:len(valid)].tobytes();assert byte_equal
        one=arrays['state_flush_False'];two=arrays['state_flush_True'];assert one.shape==two.shape and np.isfinite(one).all() and np.isfinite(two).all()
        def count(value):
            bits=value.view(np.uint32);return int((((bits&0x7f800000)==0)&((bits&0x007fffff)!=0)).sum())
        old_count,new_count=count(one),count(two);assert old_count>0 and new_count==0
        state_difference=float(abs(one.astype(np.float64)-two.astype(np.float64)).max())
    records=read(root/'records.json')
    before=[e for e in records if not e['flush_denormal']];after=[e for e in records if e['flush_denormal']]
    assert len(before)==len(after)==len(events) and 'failure' in before[-1] and all('failure' not in e for e in after)
    assert before[-1]['failure']['event']==len(valid)
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),diagnostic_result_sha256=sha256(root/'result.json'),seed=None,optimizer_updates=0,vehicle_control_sent=False))
    write_json(run/'result.json',dict(status='completed_independent_moving_native_numerics_audit',original_predictions_verified=len(valid),
        candidate_events_verified=len(events),original_prediction_max_difference=errors,prediction_arrays_byte_equal=byte_equal,
        baseline_final_subnormal_count=old_count,candidate_final_subnormal_count=new_count,final_hidden_state_max_difference=state_difference,
        hidden_states_identical=False,actor_qualified=False,vehicle_control_sent=False,
        termination_reason='Original moving output parity, raw hidden-state differences and failing decision independently checked; live timing and autonomous driving qualification remain separate'))

if __name__=='__main__':main()
