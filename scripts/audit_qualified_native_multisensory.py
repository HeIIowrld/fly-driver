"""Diagnostic native tyre/acceleration audit on complete qualified teacher laps.

Renderer poses and centred future velocity differences are EVALUATION ONLY.
Actor sidecars contain same-packet native body quantities only. Existing body21
arrays, false acceleration masks, source certificates and old models stay intact.
"""
import argparse,base64,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from flybrain_drive.ac_native_physics import PHYSICS_LAYOUT,PHYSICS_PREFIX
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
from flybrain_drive.ac_csp_geometry import matrix
from flybrain_drive.camera_body_alignment import source_render_witness
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from scripts.diagnose_sac_anticipation import load_verified,read
from scripts.sac_cohort_contract import DEFAULT_SPLIT,three_episode_cohort


def all_fields(prefix):
    values=PHYSICS_PREFIX.unpack(prefix);out={};offset=0
    for name,kind,count in PHYSICS_LAYOUT:
        v=values[offset:offset+count];out[name]=v[0] if count==1 else v;offset+=count
    return out


def native(path,expected):
    if sha256(path)!=expected:raise ValueError('Qualified native archive hash differs')
    rows=[];previous=-1;previous_digest=None;duplicates=changed=0
    with path.open(encoding='utf-8') as stream:
        for sequence,line in enumerate(stream):
            row=json.loads(line);a=row['authority'];c=row['copy'];raw=base64.b64decode(a['prefix_base64'],validate=True)
            if row['sequence']!=sequence or c.get('synthetic_replay') or hashlib.sha256(raw).hexdigest()!=a['prefix_sha256']:raise ValueError('Original continuous native authority required')
            s=decode_teacher_physics(raw,a['received_monotonic_seconds']);pid=s.physics.packet_id
            if s.builtin_ai_controlled or pid!=a['packet_id'] or pid<previous:raise ValueError('Original native identity/order differs')
            bracket=[c[k] for k in ['first_copy_start','first_copy_end','second_copy_start','second_copy_end']]
            if any(x>y for x,y in zip(bracket,bracket[1:])) or abs(bracket[-1]-a['received_monotonic_seconds'])>1e-9:raise ValueError('Native receipt bracket differs')
            digest=hashlib.sha256(raw[:PHYSICS_PREFIX.size]).hexdigest()
            if pid==previous:
                duplicates+=1;changed+=digest!=previous_digest;continue
            f=all_fields(raw[:PHYSICS_PREFIX.size]);rows.append((pid,a['received_monotonic_seconds'],f,digest))
            previous=pid;previous_digest=digest
    t=np.array([r[1] for r in rows]);assert (np.diff(t)>0).all()
    fields={key:np.asarray([r[2][key] for r in rows]) for key in rows[0][2]}
    return np.array([r[0] for r in rows]),t,fields,dict(distinct_packets=len(rows),duplicate_copies=duplicates,repeated_packet_changed_prefix_copies=int(changed),first_receipt_preserved=True)


def acceleration_metrics(a,b):
    error=a-b;std=b.std(0);corr=[]
    for i in range(3):corr.append(float(np.corrcoef(a[:,i],b[:,i])[0,1]) if a[:,i].std()>1e-10 and std[i]>1e-10 else 0.)
    rmse=np.sqrt((error**2).mean(0));pos=(b>.5).sum(0);neg=(b<-.5).sum(0)
    return dict(n=len(a),correlation=corr,rmse_m_s2=rmse.tolist(),target_std_m_s2=std.tolist(),normalized_rmse=(rmse/np.maximum(std,1e-8)).tolist(),mean_error_m_s2=error.mean(0).tolist(),positive_target_samples=pos.tolist(),negative_target_samples=neg.tolist(),candidate_axis_threshold_passed=((np.asarray(corr)>=.9)&(rmse<=.3*np.maximum(std,1e-8))&(pos>=20)&(neg>=20)).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--split',default=DEFAULT_SPLIT);a=p.parse_args()
    entries=three_episode_cohort(a.split);items=[load_verified(e) for e in entries]
    sdk=Path('C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/extension/internal/lua-sdk/ac_apps/lib.lua')
    run=begin_run(a.output,dict(arguments=vars(a),seed=780601,program_sha256=sha256(__file__),entries=entries,installed_primary_sdk=str(sdk),installed_primary_sdk_sha256=sha256(sdk),sdk_definition_lines=dict(acceleration=4052,tyre_load=3957),candidate_acceleration_indices=[2,0,1],candidate_signs=[1,1,1],g_m_s2=9.80665,
        fixed_axis_hypothesis='Prior diagnostic plus local installed SDK; no data-fitted scale/sign',diagnostic_derivative='Centred .075s world velocity slope, original receipt gaps<=20ms, >=12 points; never actor input',candidate_criteria='Each axis corr>=.9, RMSE<=.3*target_std, >=20 positive and negative >.5m/s2, >=100 matched fast frames; require all episodes; not automatic global sensor/actor qualification',actor_fields='Same original native packet accG3, wheelSlip4, wheelLoad4, tyreCoreTemperature4, wheelsPressure4; no derivative, world pose, teacher action or future observations',source_body21_unchanged=True,actor_deployed=False,vehicle_control_sent=False))
    out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,old_acceleration_masks_changed=False);start=time.monotonic();reports=[];hashes={}
    try:
        for ep,(entry,d) in enumerate(zip(entries,items)):
            cfg=read(Path(entry['data'])/'manifest.json')['config'];root=Path(cfg['arguments']['recording']);qr=read(Path(cfg['arguments']['qualification'])/'result.json')
            ids,t,f,counts=native(root/'camera/native-body.jsonl',qr['native_body_sha256']);joined=np.searchsorted(ids,d['body_packet']);np.testing.assert_array_equal(ids[joined],d['body_packet']);np.testing.assert_allclose(t[joined],d['body_receipt'],atol=1e-9,rtol=0)
            for actual,want in [(f['tyreCoreTemperature'][joined],d['body'][:,12:16]),(f['speedKmh'][joined]/3.6,d['body'][:,16]),(f['wheelAngularSpeed'][joined],d['body'][:,17:21])]:np.testing.assert_allclose(actual,want,atol=1e-6,rtol=1e-6)
            assert not d['body_valid'][:,5:8].any()
            names=['accG','wheelSlip','wheelLoad','tyreCoreTemperature','wheelsPressure'];raw=np.concatenate([f[key][joined] for key in names],axis=1).astype(np.float32)
            file=run/f'native-auxiliary-{ep}.npz';np.savez_compressed(file,source_values=raw,body_packet=d['body_packet'],body_receipt=d['body_receipt'],decision_time=d['decision_time'],fit_mask=d['fit_mask'])
            hashes[file.name]=sha256(file)
            path=root/'camera/vision';bundle=read(path/'bundle.json');catalog=read(path/'archive.json')
            if sha256(path/'camera.zip')!=qr['camera_bundle_sha256'] or bundle['sha256']!=qr['camera_bundle_sha256'] or sha256(path/'archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('Original camera bundle/catalog differs')
            catalog={x['file']:x['sha256'] for x in catalog};records=[];needed=np.unique(d['camera_index']);frames=0
            with zipfile.ZipFile(path/'camera.zip') as z:
                def member(name):
                    b=z.read(name)
                    if hashlib.sha256(b).hexdigest()!=catalog[name]:raise ValueError('Original metadata member differs')
                    return json.loads(b)
                stream=[json.loads(line) for line in z.read('stream.jsonl').decode().splitlines()]
                if hashlib.sha256(z.read('stream.jsonl')).hexdigest()!=catalog['stream.jsonl']:raise ValueError('Stream differs')
                for ci in needed:
                    frame=member(Path(stream[int(ci)]['file']).name);views={x['view_id']:x for x in frame['views']};v=views['front'];source_render_witness(v)
                    # Use physical transform, not the extrapolated camera/body pose.
                    witness=v['source_body_before_render'];transform=matrix(witness['physics_transform_diagnostic']);rotation=transform[:3,:3]
                    if not np.allclose(rotation@rotation.T,np.eye(3),atol=3e-5):raise ValueError('Physical transform is not orthonormal')
                    left=np.asarray(views['left']['renderer_only_world_look']);alignment=float(rotation[0]@left)
                    if abs(alignment)<.999:raise ValueError('Registered left view does not match physical lateral axis')
                    basis=np.stack([rotation[2],rotation[0]*np.sign(alignment),rotation[1]])
                    lower=next(x['render_started_lower'] for x in stream[int(ci)]['views'] if x['view_id']=='front');j=int(np.searchsorted(t,lower,side='left')-1)
                    if j<0 or lower-t[j]>.025:continue
                    lo=np.searchsorted(t,t[j]-.075);hi=np.searchsorted(t,t[j]+.075,side='right');tt=t[lo:hi]
                    if len(tt)<12 or tt[0]>t[j]-.06 or tt[-1]<t[j]+.06 or np.max(np.diff(tt))>.02:continue
                    dt=tt-tt.mean();velocity=f['velocity'][lo:hi];derivative=(dt[:,None]*(velocity-velocity.mean(0))).sum(0)/(dt@dt)
                    candidate=f['accG'][j][[2,0,1]]*9.80665;target=basis@derivative
                    records.append([int(ci),int(ids[j]),t[j],f['speedKmh'][j]/3.6,*candidate,*target,*np.asarray(witness['acceleration_g']),*f['accG'][j],float(alignment)])
                    frames+=1
                    if frames%200==0:write_json(run/'progress.json',dict(episode=ep,matched_frames=frames,total=len(needed),elapsed_seconds=time.monotonic()-start))
            records=np.asarray(records);fast=records[:,3]>20;stationary=(f['speedKmh']<.1);speed_error=np.linalg.norm(f['velocity'],axis=1)-f['speedKmh']/3.6
            fit=d['fit_mask'];report=dict(episode=ep,role='train' if ep<2 else 'consumed_development',**counts,qualified_actor_events=len(d['labels']),actor_join_exact=True,raw_field_groups=names,actor_raw_shape=list(raw.shape),speed_range_m_s=[float(f['speedKmh'].min()/3.6),float(f['speedKmh'].max()/3.6)],speed_world_velocity_norm_rmse_m_s=float(np.sqrt(np.mean(speed_error**2))),matched_frames=len(records),fast_frames=int(fast.sum()),acceleration_all=acceleration_metrics(records[:,4:7],records[:,7:10]),acceleration_fast=acceleration_metrics(records[fast,4:7],records[fast,7:10]),stationary_samples=int(stationary.sum()),stationary_acceleration_native_g_mean=f['accG'][stationary].mean(0).tolist() if stationary.any() else None,stationary_acceleration_native_g_max_abs=np.max(abs(f['accG'][stationary]),axis=0).tolist() if stationary.any() else None,temperature_core_native_min=f['tyreCoreTemperature'][joined][fit].min(0).tolist(),temperature_core_native_max=f['tyreCoreTemperature'][joined][fit].max(0).tolist(),slip_native_percentiles=np.percentile(f['wheelSlip'][joined][fit],[0,50,95,99,100],axis=0).tolist(),native_load_range=[float(f['wheelLoad'].min()),float(f['wheelLoad'].max())],pressure_native_range=[float(f['wheelsPressure'].min()),float(f['wheelsPressure'].max())],source_native_sha256=qr['native_body_sha256'],source_camera_sha256=qr['camera_bundle_sha256'])
            reports.append(report);file=run/f'acceleration-diagnostic-{ep}.npz';np.savez_compressed(file,records=records);hashes[file.name]=sha256(file)
            write_json(run/f'episode-{ep}.json',report)
        passed=np.all([r['acceleration_fast']['candidate_axis_threshold_passed'] for r in reports],axis=0)&np.array([r['fast_frames']>=100 for r in reports]).all()
        out.update(status='completed_qualified_native_multisensory_audit',episodes=reports,files_sha256=hashes,candidate_axes_passed_all_episodes=passed.tolist(),acceleration_body_registered=False,actor_qualified=False,training_admissible=False,sidecar_scope='Native signal recovery and exact actor-packet joins only; axis diagnostics are not automatic actor qualification',termination_reason='All three qualified raw native/camera sources, same-packet tyre extras and physical-frame derivative diagnostics audited; original masks preserved')
    except Exception as e:out.update(error=type(e).__name__,reason=str(e),termination_reason='Explicit raw multisensory audit failure')
    out['elapsed_seconds']=time.monotonic()-start;write_json(run/'result.json',out);print(json.dumps(out))
    if out['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
