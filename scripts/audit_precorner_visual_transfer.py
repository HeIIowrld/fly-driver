"""Recompute frozen steering and offline visual intervention evidence independently."""
import argparse
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.sac_cohort_contract import three_episode_cohort
from scripts.train_qualified_sac_unified import load_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(a.replay); manifest = read(root/'manifest.json')
    config = manifest['config']; result = read(root/'result.json'); plan = config['plan']
    assert result['status'] == 'completed_frozen_precorner_visual_transfer'
    assert result['optimizer_updates'] == 0 and result['vehicle_control_sent'] is False
    assert sha256(root/'executed-entrypoint.py') == plan['program_sha256']
    transfer = read(plan['transfer_plan'])
    assert sha256(plan['transfer_plan']) == plan['transfer_plan_sha256']
    assert sha256(transfer['split']) == transfer['split_sha256']
    entries = three_episode_cohort(transfer['split'])
    heads = {}
    for name in ('source','candidate'):
        if name=='candidate' and 'native' in plan:
            path=Path(plan['native']['head'])/'native-steering.pt'
            assert sha256(path)==plan['native']['head_sha256']
        else:
            path = Path(transfer[name])/'steering.pt'
            assert sha256(path) == transfer[name+'_sha256']
        heads[name] = torch.load(path,map_location='cpu',weights_only=True)['readout']
    reference_path = Path(plan['reference'])/'predictions.npz'
    assert sha256(reference_path) == plan['reference_predictions_sha256']
    native_extension=None
    if 'native' in plan:
        from flybrain_drive.audited_native_flight_candidate import load_audited_native_flight_candidate
        _,receipt=load_audited_native_flight_candidate(plan['native'])
        assert receipt==config['native_candidate']
        native_cfg=read(Path(plan['native']['fit'])/'manifest.json')['config']
        assert native_cfg['entries']==entries
        native_extension=Path(native_cfg['arguments']['extension'])
    run = begin_run(a.output, dict(replay=str(root),replay_result_sha256=sha256(root/'result.json'),
        program_sha256=sha256(__file__),optimizer_updates=0,seed=None,
        actor_qualified=False,vehicle_control_sent=False))
    reports = []; signed_all = []; single_all = []; total_events = 0; total_pre = 0
    with np.load(reference_path,allow_pickle=False) as reference:
        for episode,(entry,report) in enumerate(zip(entries,result['comparisons'],strict=True)):
            path = root/f'episode-{episode}.npz'
            assert sha256(path) == report['predictions_sha256']
            d,c,provenance = load_episode(entry['data']); t = d['decision_time']
            assert provenance['arrays_sha256'] == entry['arrays_sha256']
            with np.load(path,allow_pickle=False) as z:
                if native_extension is not None:
                    sidecar=native_extension/f'extension-{episode}.npz'
                    assert sha256(sidecar)==read(native_extension/'result.json')['files_sha256'][sidecar.name]
                    with np.load(sidecar,allow_pickle=False) as native:
                        np.testing.assert_array_equal(z['native_values'],native['values'])
                        np.testing.assert_array_equal(z['native_valid'],native['valid'])
                        for key in ('body_packet','body_receipt','decision_time'):np.testing.assert_array_equal(native[key],d[key])
                assert all(np.isfinite(z[k]).all() for k in z.files)
                normal = z['normal']; source = z['source']; pre = z['pre_mask']; muscles = z['muscles']
                np.testing.assert_allclose(normal,reference[f'candidate_{episode}'],rtol=0,atol=2e-6)
                source_key='baseline' if 'native' in plan else 'source'
                np.testing.assert_allclose(source,reference[f'{source_key}_{episode}'],rtol=0,atol=2e-6)
                assert np.array_equal(source[:,1:],normal[:,1:])
                numerical = {}; checks = []
                for name,values in [('source',source),('candidate',normal)]:
                    h = heads[name]
                    features = (muscles.astype(float)-h['training_reference'].numpy())/h['training_scale'].numpy()
                    logits = features@h['projection.weight'].numpy().astype(float).T+h['projection.bias'].numpy()
                    predicted = np.column_stack([np.tanh(logits[:,0]),1/(1+np.exp(-logits[:,1:]))])
                    error = float(np.max(abs(predicted-values)))
                    assert error < 2e-6
                    numerical[name] = error
                    for key,mask in [('all_rmse',d['fit_mask']),('pre_rmse',pre)]:
                        assert mask.any()
                        value = float(np.sqrt(np.mean((values[mask,0]-d['labels'][mask,0])**2)))
                        assert abs(value-report[name][key]) < 1e-12
                all_pre = np.zeros(len(t),dtype=bool); indices_seen = set()
                aggregate = {name:[] for name in ('normal','spatial_neutral')}
                branch_abs = {name:[] for name in ('opposite','delayed600','frozen_start','spatial_neutral')}
                for wi,w in enumerate(report['windows']):
                    ids = z[f'window_{wi}_receiver_indices']; mask = z[f'window_{wi}_pre_mask']
                    assert np.array_equal(ids,np.arange(w['start'],w['event']+1))
                    assert len(set(ids[mask]) & indices_seen) == 0
                    indices_seen.update(ids[mask].tolist()); all_pre[ids[mask]] = True
                    expected_mask = pre[ids] & (t[ids]>=w['time']-.8) & (t[ids]<=w['time']-.08)
                    np.testing.assert_array_equal(mask,expected_mask)
                    assert np.all(abs(d['body'][ids[mask],4]) < .07)
                    np.testing.assert_allclose(z[f'window_{wi}_normal'],normal[ids],rtol=0,atol=2e-6)
                    branch = {}
                    names = ['normal','delayed600','frozen_start','spatial_neutral']
                    if w['donor_time'] is not None: names.append('opposite')
                    for name in names:
                        j = z[f'window_{wi}_{name}_indices']
                        if name in ('normal','spatial_neutral'): expected = ids
                        elif name == 'frozen_start': expected = np.full(len(ids),ids[0])
                        else:
                            wanted = t[ids]-.6 if name=='delayed600' else t[ids]-w['time']+w['donor_time']
                            expected = np.searchsorted(t,wanted,side='right')-1
                            assert np.all(expected>=0) and np.all(t[expected]<=wanted)
                        np.testing.assert_array_equal(j,expected)
                        values = z[f'window_{wi}_{name}']
                        if name!='normal' and mask.any():
                            changes = abs(values[mask,0]-normal[ids[mask],0])
                            branch_abs[name].extend(changes.tolist()); branch[name] = float(changes.mean())
                            assert abs(branch[name]-w['mean_abs_steering_change'][name]) < 1e-12
                        if name in aggregate: aggregate[name].extend(values[mask,0].tolist())
                    signed = None
                    if 'opposite' in names and mask.any():
                        signed = float(np.mean((normal[ids[mask],0]-z[f'window_{wi}_opposite'][mask,0])*w['direction']))
                        assert abs(signed-w['signed_normal_minus_opposite']) < 1e-12
                        signed_all.append(signed)
                        for i in ids[mask]:
                            instant = z[f'instant_{i}']; j = int(instant[0])
                            expected = np.searchsorted(t,t[i]-w['time']+w['donor_time'],side='right')-1
                            assert j == expected and instant[1] == normal[i,0]
                            single_all.append(dict(episode=episode,event=int(i),
                                absolute_change=float(abs(instant[1]-instant[2])),
                                signed_change=float((instant[1]-instant[2])*w['direction'])))
                    checks.append(dict(event=w['event'],count=int(mask.sum()),
                        signed_normal_minus_opposite=signed,mean_absolute_change=branch))
                np.testing.assert_array_equal(all_pre,pre)
                labels = d['labels'][np.array(sorted(indices_seen)),0]
                causal = {}
                # Window order follows strictly increasing events; same original labels.
                for name,values in aggregate.items():
                    causal[name] = float(np.sqrt(np.mean((np.asarray(values)-labels)**2)))
                reports.append(dict(episode=episode,model_events=len(t),pre_count=int(pre.sum()),
                    source=report['source'],candidate=report['candidate'],windows=checks,
                    independent_head_max_error=numerical,
                    spatial_path_precorner_rmse=causal,
                    mean_absolute_changes={k:float(np.mean(v)) if v else None for k,v in branch_abs.items()}))
                total_events += len(t); total_pre += int(pre.sum())
    out = dict(status='completed_independent_precorner_visual_transfer_audit',
        verified_model_events=total_events,verified_pre_labels=total_pre,
        expected_direction_windows=sum(v>0 for v in signed_all),paired_windows=len(signed_all),
        mean_signed_window_effect=float(np.mean(signed_all)) if signed_all else None,
        instantaneous_labels=len(single_all),
        instantaneous_mean_absolute_change=float(np.mean([v['absolute_change'] for v in single_all])) if single_all else None,
        instantaneous_mean_signed_change=float(np.mean([v['signed_change'] for v in single_all])) if single_all else None,
        comparisons=reports,native_candidate=config.get('native_candidate'),optimizer_updates=0,actor_qualified=False,vehicle_control_sent=False,
        limitations=['Same Monza corners repeated across recordings; windows are not independent road geometries',
            'Feature boundary intervention, no pixel/FlyVis re-simulation or physical counterfactual action truth',
            'Timing of yaw-transition proxy is not surveyed geometric corner entry',
            'No autonomous student lap or unseen-track qualification'],
        termination_reason='Raw array hashes, original full replay, independent muscle/head formula, receiver masks, donor floor times and saved intervention metrics verified')
    write_json(run/'instantaneous-effects.json',single_all); write_json(run/'result.json',out)


if __name__=='__main__': main()
