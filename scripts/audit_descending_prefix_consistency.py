"""Independent per-axis TRAIN prefix discrepancy and RMSE audit from raw arrays."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode
from scripts.fit_bounded_sac_steering import pre_mask

def main():
 root=Path('runs/descending-visual-intent-20261104');source=root/'prefix-consistency-01';r=read(source/'result.json');assert r['status']=='completed_training_prefix_consistency_diagnostic' and sha256(source/'predictions.npz')==r['predictions_sha256'];m=read(root/'real_only-fit-01/manifest.json')['config'];original=Path(read(m['plan']['candidate_plan'])['motion']['fit']);episodes={e:load_episode(m['entries'][e]['data'])[0] for e in (0,1)};tasks=m['real_windows'];target=np.concatenate([episodes[t['episode']]['labels'][t['begin']:t['end']] for t in tasks]);fit=np.concatenate([episodes[t['episode']]['fit_mask'][t['begin']:t['end']] for t in tasks]);pre=np.concatenate([pre_mask(episodes[t['episode']],-1)[t['begin']:t['end']] for t in tasks]);reports=[]
 with np.load(source/'predictions.npz') as z:
  for report in r['reports']:
   arm=report['model'];p=z[arm+'_fixed_prefix'];q=z[arm+'_continuous'];full={};hashes={}
   for e in (0,1):
    file=original/'predictions.npz' if arm=='parent' else root/f'actual-{arm}-{e}-01/predictions.npz'
    with np.load(file) as data:full[e]=data[f'prediction_{e}' if arm=='parent' else 'prediction'].copy()
    hashes[str(file)]=sha256(file)
   expected=np.concatenate([full[t['episode']][t['begin']:t['end']] for t in tasks]);np.testing.assert_array_equal(q,expected);difference=abs(p-q);assert float(difference.max())==report['maximum_prediction_difference'];metrics={}
   for role,mask in [('all_selected_fit',fit),('pre',pre)]:
    for name,values in [('fixed_parent_prefix_rmse',p),('continuous_candidate_rmse',q)]:np.testing.assert_allclose(np.sqrt(np.mean((values[mask].astype(float)-target[mask])**2,axis=0)),report['metrics'][role][name],atol=1e-12,rtol=0)
    metrics[role]=dict(maximum_axis_difference=difference[mask].max(0).tolist(),rms_axis_difference=np.sqrt(np.mean(difference[mask].astype(float)**2,axis=0)).tolist())
   reports.append(dict(model=arm,events=len(p),maximum_axis_difference=difference.max(0).tolist(),metrics=metrics,continuous_source_sha256=hashes))
 write_json(root/'prefix-consistency-audit-01.json',dict(status='completed_independent_prefix_consistency_statistics',source_result_sha256=sha256(source/'result.json'),source_predictions_sha256=sha256(source/'predictions.npz'),program_sha256=sha256(__file__),reports=reports,full_neural_replay_by_audit=False,vehicle_control_sent=False,optimizer_updates=0))
if __name__=='__main__':main()
