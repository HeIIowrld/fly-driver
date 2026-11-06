"""Fit local high-speed corrections while preserving the sealed phase head at launch anchors."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from flybrain_drive.compact_all_muscle_readout import CompactAllMuscleReadout, all_muscle_features
from flybrain_drive.compact_flight_pedal_readout import CompactFlightPedalReadout, compact_features_np
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.joint_balance_muscle_readout import JointBalanceMusclePedalReadout
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.sac_unified_student import read
from scripts.audit_flight_braking_phase_errors import phase_masks
from scripts.fit_current_flight_launch_pedals import datasets, metrics


def bases(points, anchors, maximum=40):
    distance = np.sqrt(((points[:,None,:]-anchors)**2).sum(-1)).min(1)
    pool = np.flatnonzero(distance > .08)
    if len(pool) < 8:
        raise ValueError("No distinct high-speed balance support")
    # Farthest-point coverage within genuine high-speed TRAIN rows.
    selected, min_distance = [], np.full(len(points), np.inf)
    j = int(pool[np.argmax(distance[pool])])
    while len(selected) < maximum:
        selected.append(j)
        min_distance = np.minimum(min_distance, ((points-points[j])**2).sum(-1))
        j = int(pool[np.argmax(min_distance[pool])])
        if min_distance[j] < .0064:
            break
    centers = points[selected]
    separation = np.sqrt(((centers[:,None,:]-centers)**2).sum(-1))
    np.fill_diagonal(separation,np.inf)
    radii = np.minimum(.95*distance[selected], 2.2*separation.min(1))
    return centers.astype(np.float32), radii.astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-fit", required=True)
    p.add_argument("--launch-witness", required=True)
    p.add_argument("--additional-launch-witness", action="append", default=[])
    p.add_argument("--protect-compact", type=Path)
    p.add_argument("--inactive-label-weight", type=float, choices=(0., .5, 1.), default=0.)
    p.add_argument("--isolated-pedal-objective", action="store_true",
                   help="Protect each pedal within both one-pedal TRAIN conditions")
    p.add_argument("--all-muscle-residual", action="store_true",
                   help="Fit a six-muscle residual while freezing the protected compact head")
    p.add_argument("--output", required=True)
    p.add_argument("--max-iterations", type=int, choices=(400,800), default=400)
    a=p.parse_args()
    protected = a.protect_compact
    a.protect_compact = str(protected) if protected else None
    if a.isolated_pedal_objective and (not protected or a.inactive_label_weight):
        raise ValueError("Isolated-pedal fit requires a protected compact and its own objective")
    if a.all_muscle_residual and not a.isolated_pedal_objective:
        raise ValueError("Six-muscle residual requires protected one-pedal TRAIN groups")
    torch.set_num_threads(1)
    torch.manual_seed(784107)
    source=Path(a.phase_fit)
    cfg=read(source/"manifest.json")["config"]
    source_result=read(source/"result.json")
    if (source_result["status"]!="completed_balanced_muscle_pedal_fit"
            or sha256(source/"pedals.pt")!=source_result["checkpoint_sha256"]):
        raise ValueError("Exact completed phase fit required")
    out=begin_run(a.output,dict(arguments=vars(a),phase_checkpoint_sha256=source_result["checkpoint_sha256"],
        launch_witness_sha256=sha256(a.launch_witness),
        additional_launch_witness_sha256={path:sha256(path) for path in a.additional_launch_witness},
        protected_checkpoint_sha256=sha256(protected/"pedals.pt") if protected else None,
        program_sha256=sha256(__file__),
        compact_readout_sha256=sha256("src/flybrain_drive/compact_flight_pedal_readout.py"),
        all_muscle_readout_sha256=sha256("src/flybrain_drive/compact_all_muscle_readout.py")
            if a.all_muscle_residual else None,
        phase_group_source_sha256=sha256("scripts/audit_flight_braking_phase_errors.py"),
        seed=784107,development_used_for_fit=False,vehicle_control_sent=False,actor_qualified=False))
    result=dict(status="failed",vehicle_control_sent=False,actor_qualified=False)
    try:
        cp=torch.load(source/"pedals.pt",map_location="cpu",weights_only=True)["state"]
        parent,_=load_muscle_interaction_head(cfg["fixed_decoder"])
        phase=JointBalanceMusclePedalReadout(parent,cp["centers"],cp["width"])
        phase.offset.load_state_dict(cp["offset"])
        phase.pedal_projection.load_state_dict(cp["pedal_projection"])
        phase.verify()
        data=datasets(Path(cfg["evaluation"]))
        train=data[:2]
        if [item["receipt"]["role"] for item in train]!=["train","train"]:
            raise ValueError("Two original TRAIN episodes required")
        ref,scale=(getattr(parent.base,"training_"+key).detach().numpy() for key in ("reference","scale"))
        muscle=np.concatenate([v["muscles"] for v in train])
        z=((muscle-ref)/scale).astype(np.float64)
        observed=[]
        for path in [a.launch_witness,*a.additional_launch_witness]:
            with np.load(path,allow_pickle=False) as witness:
                values=witness["normal_muscles"]
            if values.ndim!=2 or values.shape[1]!=6 or len(values)<24 or not np.isfinite(values).all():
                raise ValueError("Real replayed launch muscle witness required")
            observed.append(values)
        observed=np.concatenate(observed)
        low=np.concatenate([v["low"] for v in train])
        high=np.concatenate([v["high"] for v in train])
        dims=6 if a.all_muscle_residual else 4
        features=all_muscle_features if a.all_muscle_residual else compact_features_np
        anchors=np.concatenate([z[low,:dims],((observed-ref)/scale)[:,:dims]])
        if protected:
            proof=read(protected/"result.json")
            checkpoint=protected/"pedals.pt"
            if proof["status"]!="completed_compact_braking_fit" or sha256(checkpoint)!=proof["checkpoint_sha256"]:
                raise ValueError("Exact completed compact reference required")
            prior=torch.load(checkpoint,map_location="cpu",weights_only=True)
            if (prior["actor_kind"]!=cfg["actor_kind"] or
                    prior["phase_checkpoint_sha256"]!=source_result["checkpoint_sha256"]):
                raise ValueError("Protected compact lineage differs")
            if prior["schema"]!="flybrain.compact_flight_pedals.experimental.v1":
                raise ValueError("Protect the sealed single compact head")
            if a.all_muscle_residual:
                centers,radii=bases(z[high],anchors)
                initial=np.zeros(2*len(centers))
            else:
                centers,radii=prior["centers"].numpy(),prior["radii"].numpy()
                initial=prior["coefficients"].numpy().astype(float).ravel()
        else:
            centers,radii=bases(z[high,:4],anchors)
            initial=np.zeros(2*len(centers))
        frozen=np.any(features(anchors,centers,radii)!=0,axis=0)
        if not protected and frozen.any():
            raise ValueError("Compact correction overlaps exact launch anchors")
        if protected:
            with torch.no_grad():
                anchor_muscles=torch.tensor(np.concatenate((muscle[low],observed)))
                z_anchors=(anchor_muscles-parent.base.training_reference)/parent.base.training_scale
                q=((z_anchors[:,None,:dims]-torch.from_numpy(centers))/torch.from_numpy(radii)[None,:,None]).square().sum(-1)
                frozen|=(q<1).any(0).numpy()
        bounds=[(float(v),float(v)) if lock else (-8,8) for v,lock in zip(initial,np.tile(frozen,2))]
        # Fit only the original high labels. Launch rows are exact geometric exclusions.
        x=features(z[high],centers,radii)
        label=np.concatenate([v["data"]["labels"][:,1:] for v in train])[high].astype(float)
        with torch.no_grad():
            mu=torch.tensor(muscle[high])
            base=(phase.pedal_projection(torch.tensor(z[high],dtype=torch.float32))[:,1:]
                  +phase.offset(phase.features(mu))).numpy().astype(float)
        if a.all_muscle_residual:
            base+=compact_features_np(z[high],prior["centers"].numpy(),prior["radii"].numpy()) @ prior["coefficients"].numpy().T
        baseline=expit(base+x@initial.reshape(2,len(centers)).T)
        groups=[]; offset=0
        for i,item in enumerate(train):
            masks,_,_=phase_masks(item)
            if a.isolated_pedal_objective:
                throttle, brake = item["data"]["labels"][:,1:].T
                isolated = {
                    "teacher_brake_only": item["high"] & (throttle<=.02) & (brake>=.1),
                    "teacher_throttle_only": item["high"] & (brake<=.02) & (throttle>=.1),
                }
                if any(np.count_nonzero(mask)<8 for mask in isolated.values()):
                    raise ValueError("Both one-pedal TRAIN conditions need independent support")
                masks.update(isolated)
            for name,mask in masks.items():
                local=np.flatnonzero(mask[item["high"]])
                if len(local)>=8:
                    groups.append((i,name,local+offset))
            offset+=int(item["high"].sum())
        limits=np.array([np.mean((baseline[ix]-label[ix])**2,axis=0) for _,_,ix in groups])
        weights=np.zeros(len(label))
        for _,_,ix in groups:
            weights[ix]+=1/(len(groups)*len(ix))
        n=len(centers)
        # On TRAIN only, a teacher actively using the other pedal gives
        # stronger evidence that this near-zero label matters.
        inactive=(label<=.02)&(label[:,::-1]>=.1)
        output_weights=weights[:,None]*(1+a.inactive_label_weight*inactive)
        if a.isolated_pedal_objective:
            # Each teacher-quiet pedal contributes equally despite different label counts.
            # Protected per-condition constraints still cover both pedal errors.
            output_weights = .25*weights[:,None] + .75*inactive/inactive.sum(axis=0)
        def pred(v):return expit(base+x@v.reshape(2,n).T)
        def obj(v):
            y=pred(v);e=y-label
            return np.sum(output_weights*e**2)/2+1e-5*np.mean(v**2), (
                (e*y*(1-y)*output_weights).T@x).ravel()+2e-5*v/len(v)
        def constraint(v):
            e=pred(v)-label
            return (limits-np.array([np.mean(e[ix]**2,axis=0) for _,_,ix in groups])).ravel()
        def jac(v):
            y=pred(v);slope=2*(y-label)*y*(1-y)
            g=np.zeros((2*len(groups),2,n))
            for i,(_,_,ix) in enumerate(groups):
                grad=-(slope[ix].T@x[ix])/len(ix)
                g[2*i,0],g[2*i+1,1]=grad
            return g.reshape(2*len(groups),2*n)
        journal=[]
        def record(v):journal.append(dict(loss=float(obj(v)[0]),minimum_slack=float(constraint(v).min()),coefficients=v.tolist()))
        opt=minimize(obj,initial,method="SLSQP",jac=True,bounds=bounds,
            constraints=[dict(type="ineq",fun=constraint,jac=jac)],callback=record,
            options=dict(maxiter=a.max_iterations,ftol=1e-10))
        coeff=torch.tensor(opt.x.reshape(2,n),dtype=torch.float32)
        compact=(CompactFlightPedalReadout(phase,prior["centers"],prior["radii"],prior["coefficients"])
                 if protected else phase)
        head=(CompactAllMuscleReadout(compact,torch.tensor(centers),torch.tensor(radii),coeff)
              if a.all_muscle_residual else CompactFlightPedalReadout(phase,torch.tensor(centers),torch.tensor(radii),coeff))
        rows,output=metrics(head,data)
        with torch.no_grad():
            anchors_tensor=torch.tensor(np.concatenate((muscle[low],observed)))
            anchor_error=float((head(anchors_tensor)-compact(anchors_tensor)).abs().max())
        f32=np.concatenate([output[f"prediction_{i}"][item["high"],1:] for i,item in enumerate(train)])
        ratios=[(np.mean((f32[ix]-label[ix])**2,axis=0)/limits[i]).tolist()
                for i,(_,_,ix) in enumerate(groups)]
        feasible=(opt.success and np.max(ratios)<=1.00001 and anchor_error==0
                  and np.max(np.abs(coeff.numpy().ravel()-initial))>1e-6)
        def inactive_rmse(values):
            return [float(np.sqrt(np.mean((values[mask,axis]-label[mask,axis])**2)))
                    if mask.any() else None for axis,mask in enumerate(inactive.T)]
        checkpoint=dict(schema="flybrain.compact_flight_pedals.experimental.v1",phase_checkpoint=str(source/"pedals.pt"),
            phase_checkpoint_sha256=source_result["checkpoint_sha256"],centers=torch.tensor(centers),
            radii=torch.tensor(radii),coefficients=coeff,actor_kind=cfg["actor_kind"],actor_qualified=False)
        if a.all_muscle_residual:
            checkpoint.update(schema="flybrain.compact_flight_pedals.experimental.v2",
                prior={key:prior[key] for key in ("centers","radii","coefficients")})
        torch.save(checkpoint,out/"pedals.pt")
        np.savez_compressed(out/"predictions.npz",**output)
        write_json(out/"training-journal.json",journal)
        result.update(status="completed_compact_braking_fit" if feasible else "failed_constraints",
            optimizer_success=bool(opt.success),optimizer_message=str(opt.message),iterations=int(opt.nit),
            anchor_count=len(anchors),anchor_max_difference=anchor_error,centers=n,frozen_basis_count=int(frozen.sum()),
            high_rows_with_nonzero_support=int(np.count_nonzero(np.max(x,axis=1))),
            inactive_teacher_label_counts=inactive.sum(axis=0).tolist(),
            inactive_label_rmse_before=inactive_rmse(baseline),
            inactive_label_rmse_after=inactive_rmse(f32),
            metrics=rows,groups=[dict(episode=i,name=name,labels=len(ix),mse_ratio_to_protected_reference=ratio, **({} if protected else {"mse_ratio_to_phase":ratio}))
              for (i,name,ix),ratio in zip(groups,ratios)],checkpoint_sha256=sha256(out/"pedals.pt"),
            predictions_sha256=sha256(out/"predictions.npz"))
    except Exception as error:
        result["reason"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(out/"result.json",result)


if __name__=="__main__":
    main()
