"""Independent identities, old-motion parity and orthogonal-energy audit."""
import argparse
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recovery',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.recovery);r=read(root/'result.json');m=read(root/'manifest.json')['config']
    if r['status']!='completed_qualified_motion_spatial_recovery' or sha256(root/'motion-spatial.npz')!=r['features_sha256'] or sha256(root/'executed-entrypoint.py')!=m['program_sha256']:raise ValueError('Completed source-bound motion recovery required')
    actor=Path(m['arguments']['actor_data']);d,contract,provenance=load_episode(actor);assert provenance['arrays_sha256']==m['actor_arrays_sha256'] and sha256(actor/'result.json')==m['actor_result_sha256']
    spatial=Path(m['arguments']['spatial']);geometry_file=spatial/'pool-geometry.npz';assert sha256(geometry_file)==m['geometry_sha256']
    with np.load(geometry_file) as z:old=z['old'];inverse=z['inverse'];fine=z['grid8_narrow']
    with np.load(root/'motion-spatial.npz') as z:
        ids=z['camera_indices'];joined=z['event_to_camera'];motion=z['original_motion_pool'].reshape(-1,4,8,16);features=z['discarded_motion_grid8'];n=len(ids)
        assert features.shape==(n,2048) and np.isfinite(features).all();np.testing.assert_array_equal(ids[joined],d['camera_index']);expected=np.asarray([d['scene'][np.flatnonzero(d['camera_index']==i)[0]].reshape(4,9,16)[:,1:] for i in ids]);difference=float(abs(motion-expected).max());assert difference<=2e-6
        mean=z['mean_motion_tanh'];mean_res=z['mean_residual'];mean_null=float(abs(mean_res@old.T).max());assert mean_null<=1e-6
        np.testing.assert_allclose(features.mean(0), (mean_res@fine.T).reshape(-1),atol=2e-6,rtol=1e-5)
        row_energy=[]
        for start in range(0,n,100):row_energy.append(np.sum((motion[start:start+100].astype(float)@inverse.T)**2,axis=-1))
        row_energy=np.concatenate(row_energy);total=z['per_frame_energy'];lost=z['per_frame_lost_energy'];np.testing.assert_allclose(total,row_energy+lost,atol=2e-5,rtol=1e-5)
        variation=np.maximum(total.sum(0)-n*np.sum(mean**2,axis=-1),0);variation_lost=np.maximum(lost.sum(0)-n*np.sum(mean_res**2,axis=-1),0);valid=variation>1e-12;fraction=np.divide(variation_lost,variation,out=np.zeros_like(variation),where=valid)
        np.testing.assert_allclose(variation,z['temporal_variation_energy'],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(variation_lost,z['temporal_variation_lost_energy'],atol=1e-9,rtol=1e-9);np.testing.assert_allclose(fraction,r['temporal_variation_fraction_lost'],atol=1e-9,rtol=0);np.testing.assert_array_equal(valid,r['temporal_variation_fraction_valid']);weighted=float(variation_lost.sum()/variation.sum());assert abs(weighted-r['weighted_temporal_variation_fraction_lost'])<1e-12
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),recovery_result_sha256=sha256(root/'result.json'),seed=None,optimizer_updates=0,vehicle_control_sent=False))
    write_json(out/'result.json',dict(status='completed_independent_qualified_motion_spatial_audit',camera_groups=n,model_events=len(d['labels']),coarse_motion_difference_from_original_scene=difference,mean_residual_old_pool_max_difference=mean_null,orthogonal_per_frame_energy_identity_verified=True,weighted_temporal_variation_fraction_lost=weighted,actor_qualified=False,connected_to_actor=False,vehicle_control_sent=False,termination_reason='Original event/camera joins, old modeled motion parity, residual mean/fine-pool relation and per-frame orthogonal-energy/temporal-variation statistics independently checked; no full independent raw-neural replay, physiological validity or driving-information fraction claim.'))

if __name__=='__main__':main()
