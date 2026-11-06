"""Seal TRAIN-only direct visual-port reference and matched current-Adam continuation."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import torch
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import write_json
from flybrain_drive.sac_unified_student import read
from scripts.fixture_adjoint_speed_experiment import FixtureAdjointSpeedExperiment
ROOT=Path('runs/direct-scene-gain-learning-20261105');PARENT=Path('runs/fixture-adjoint-learning-20261104')

def main():
 torch.set_num_threads(1);ROOT.mkdir(exist_ok=False);p=read(PARENT/'plan-01.json');e=FixtureAdjointSpeedExperiment(p,fixture_input_adjoint=True)
 try:
  values=[];indices=[]
  with torch.no_grad():
   for episode in [0,1]:
    d=e.items[episode][0];ix=np.flatnonzero(d['fit_mask']);indices.append(dict(episode=episode,indices=ix.tolist()));scene=torch.tensor(d['scene'][ix]);x=torch.cat([scene.new_zeros((len(scene),12)),scene],1);values.append(torch.nn.functional.linear(x,e.model.visual.weight,e.model.visual.bias))
   reference=torch.cat(values).mean(0);count=sum(len(v) for v in values);assert count==5659 and reference.numel()==len(e.model.motor.backbone.vision_ports)
  record=dict(reference=reference.tolist(),TRAIN_decisions=count,source_weight_sha256=hashlib.sha256(e.model.visual.weight.detach().numpy().tobytes()).hexdigest(),source_bias_sha256=hashlib.sha256(e.model.visual.bias.detach().numpy().tobytes()).hexdigest(),TRAIN_indices=indices,development_used=False);write_json(ROOT/'direct-reference-01.json',record)
 finally:e.close()
 cp=PARENT/'input_adjoint-fit-01/ports.pt';adam=cp.parent/'optimizer-state.pt';r=read(cp.parent/'result.json');assert sha256(cp)==r['checkpoint_sha256'] and sha256(adam)==r['optimizer_checkpoint_sha256'];p.update(direct_parent_checkpoint=str(cp),direct_parent_checkpoint_sha256=sha256(cp),direct_parent_optimizer=str(adam),direct_parent_optimizer_sha256=sha256(adam),direct_parent_evaluation=str(PARENT/'input_adjoint-evaluation-01'),direct_initial_optimizer_steps=r['final_optimizer_steps'],initial_optimizer_steps=r['final_optimizer_steps']+[0],direct_gain_reference=reference.tolist(),direct_gain_reference_sha256=sha256(ROOT/'direct-reference-01.json'),direct_gain_learning_rate=.001,direct_gain_step_norm=.002,seed=782011,reserved_seed=915320293,reserved_plan=str(PARENT/'next-reserved-plan-01.json'),reserved_plan_sha256=sha256(PARENT/'next-reserved-plan-01.json'),scope='Same current input_adjoint and exact4 Adam groups1975/1655/662/662, new direct visual-port gain starts0. Frozen comparator and learned gain share all data/objectives/schedule/seed; new coefficient group0. Existing static routing stays frozen because TRAIN factorial showed it helps. No recurrent numeric resets within real episodes; original cumulative bounds preserved.')
 p['source_sha256']={f:sha256(f) for f in ['src/flybrain_drive/direct_scene_port_gain.py','scripts/direct_scene_gain_experiment.py','scripts/fit_direct_scene_gain.py','scripts/evaluate_direct_scene_gain.py','scripts/prepare_direct_scene_gain.py','scripts/validate_direct_scene_gain.py']};shutil.copyfile(PARENT/'normalization-01.json',ROOT/'normalization-01.json');assert sha256(ROOT/'normalization-01.json')==p['normalization_sha256'];write_json(ROOT/'plan-01.json',p)

if __name__=='__main__':main()
