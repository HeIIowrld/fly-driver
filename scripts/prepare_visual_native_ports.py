"""Declare port learning, unchanged decoder, fresh image reservation and preservation."""
from pathlib import Path
import numpy as np
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.paired_decoder_data import load_data

def main():
 root=Path('runs/visual-native-port-learning-20261104');root.mkdir(exist_ok=False);previous=read('runs/multicorner-command-20261104/source-receipt-final-01.json');old=dict(previous['preserved_production_and_checkpoint_sha256']);old.update(previous['current_checkpoints_sha256']);sources=dict(previous['preserved_prior_experiment_source_sha256']);sources.update({p:h for p,h in previous['source_and_documents_sha256'].items() if p.startswith(('src/','scripts/','tests/'))});assert all(sha256(p)==h for p,h in old.items());assert all(sha256(p)==h for p,h in sources.items());write_json(root/'initial-preservation-01.json',dict(source_revision=previous['source_revision'],preserved_production_and_checkpoint_sha256=old,preserved_prior_experiment_source_sha256=sources));mc=read('runs/multicorner-command-20261104/protocol-01.json');parent=mc['parent_checkpoint'];decoder='runs/multicorner-command-20261104/affine-fit-01/decoder.pt';_,items,_,_,pre=load_data(read(mc['data_plan']));tasks=[v for v in mc['tasks'] if v['episode']<2];selected=[]
 for episode in [0,1]:
  n=len(items[episode][0]['scene']);ids=set(range(0,(n+31)//32,4));ids.update(np.unique(np.flatnonzero(pre[episode])//32).tolist())
  for task in [v for v in tasks if v['episode']==episode]:ids.update(range(task['begin']//32,(task['end']-1)//32+1))
  selected.extend(dict(episode=episode,begin=32*b,end=min(32*b+32,n)) for b in sorted(ids))
 image_tasks=[]
 for task in [v for v in tasks if v['phase']=='early']:
  episode=task['episode'];body=items[episode][0]['body'];speed=float(body[task['begin']:task['end'],16].mean());donors=[v for v in tasks if v['episode']==episode and v['phase']=='early'];pair={}
  for name,sign in [('left_donor',-1),('right_donor',1)]:
   candidates=[v for v in donors if v['onset']['direction']==sign];donor=min(candidates,key=lambda v:(abs(float(body[v['begin']:v['end'],16].mean())-speed),v['begin']));pair[name]=donor['begin']
  image_tasks.append(dict(task,**pair))
 reserved='runs/multicorner-command-20261104/protected-reserved-plan-01.json';plan=dict(seed=781307,parent_checkpoint=parent,parent_checkpoint_sha256=sha256(parent),parent_fit_plan=mc['parent_fit_plan'],fixed_decoder=decoder,fixed_decoder_sha256=sha256(decoder),multicorner_protocol_sha256=sha256('runs/multicorner-command-20261104/protocol-01.json'),physical_tasks=tasks,image_tasks=image_tasks,image_target_margin=.002,image_direction_weight=1000.,image_anchor_weight=.5,selected_real_windows=selected,window=32,epochs=1,visual_learning_rate=.001,native_learning_rate=.001,visual_step_norm=.001,visual_total_norm=.1,visual_absolute_bound=.08,native_step_norm=.00025,native_total_norm=.01,native_entry_delta=.002,physical_weight=1000.,minimum_yaw_logit_slope=.005,minimum_speed_response_slope=.005,synthetic_native_gradient=False,image_exchange_native_gradient=False,real_only_native_optimizer_steps=True,synthetic_updates=54,reserved_plan=reserved,reserved_plan_sha256=sha256(reserved),reserved_seed=915320087,development_used_for_fit=False,source_sha256={p:sha256(p) for p in ['scripts/visual_native_port_experiment.py','scripts/prepare_visual_native_ports.py','scripts/validate_visual_native_ports.py','scripts/fit_visual_native_ports.py','scripts/evaluate_visual_native_ports.py','tests/test_port_weight_projection.py']},scope='Existing visual-current calibration and optional existing native19 port projection, fixed bidirectionally tested affine muscle decoder. Original measured chemical topology, legacy body path and all other visual weights fixed. Own continuous numerical TRAIN history; physical counterfactual cuts explicit. Actual image exchange is a retimed diagnostic augmentation, not counterfactual vehicle action truth; native projection frozen during all image/body fixtures. No actor authority.');write_json(root/'plan-01.json',plan);print(len(selected),len(tasks),len(image_tasks),len(old),len(sources))
if __name__=='__main__':main()
