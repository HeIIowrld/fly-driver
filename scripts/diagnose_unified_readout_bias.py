"""Decompose observed readout bias on recorded, fit-forbidden state histories."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_maneuver import PhysicalManeuverCircuit
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();sources=[B/'20261101-unified-stationary-path-ablations',B/'20261101-unified-moving-teacher-path-ablations']
 for root in sources:
  r=read(root/'result.json')
  if not r['status'].startswith('completed_') or sha256(root/'ablations.npz')!=r['output_sha256'] or r['training_admissible'] is not False:raise ValueError('Exact fit-forbidden ablation recording required')
 plan=read(B/'20261030-t1913-b2-whole-history-load/firstfit-b2-interface/fit-plan-SEALED.json')['model'];checkpoint=Path(plan['checkpoint']);run=begin_run(a.output,dict(program_sha256=sha256(__file__),checkpoint_sha256=sha256(checkpoint),source_hashes=[sha256(x/'result.json') for x in sources],scope='readout arithmetic diagnostic only; no fitting',optimizer_updates=0,actor=False))
 result={'status':'failed','training_admissible':False,'optimizer_updates':0,'vehicle_control_sent':False,'actor_deployed':False}
 try:
  torch.set_num_threads(1);motor=PhysicalManeuverCircuit(plan['spec'],torch.load(checkpoint,map_location='cpu',weights_only=True),sparse_backend='cached_csr_cpu').eval().requires_grad_(False);backbone=motor.backbone;readout=backbone.readout
  reference=readout.lateral_reference.detach().numpy().copy();mix=readout.steering_mix.softmax(0).detach().numpy();gain=float(readout.turn_log_gain.exp().item());reports=[];archive={}
  for root in sources:
   with np.load(root/'ablations.npz',allow_pickle=False) as z:state=z['observed_state'].copy();intent=z['observed_intent'].copy()
   with torch.no_grad():pre,muscles=backbone._read_state_preactivation_tensor(torch.from_numpy(state));decoded=pre.tanh().numpy();pre=pre.numpy();muscles=muscles.numpy()
   # Stationary run stores priming state too; moving diagnostic stores every decision.
   if len(state)==len(intent)+1:aligned=decoded[1:]
   elif len(state)==len(intent):aligned=decoded
   else:raise ValueError('Unexpected state/output accounting')
   error=float(np.max(np.abs(aligned-intent)))
   if error>2e-6:raise ValueError('Original readout mismatch')
   lateral=np.stack((muscles[:,1]-muscles[:,0],muscles[:,3]-muscles[:,2]),axis=1);contribution=(lateral-reference)*mix*gain
   np.testing.assert_allclose(contribution.sum(axis=1),pre[:,0],atol=2e-6,rtol=0)
   quant=lambda x:np.quantile(x,[0,.5,1]).tolist()
   reports.append({'source':root.name,'states':len(state),'intent_reconstruction_max_abs':error,'muscle_mean':muscles.mean(axis=0).tolist(),'lateral_reference':reference.tolist(),'mix':mix.tolist(),'gain':gain,'lateral_b1_quantiles':quant(lateral[:,0]),'lateral_b2_quantiles':quant(lateral[:,1]),'weighted_b1_turn_preactivation_mean':float(contribution[:,0].mean()),'weighted_b2_turn_preactivation_mean':float(contribution[:,1].mean()),'turn_preactivation_quantiles':quant(pre[:,0]),'turn_intent_quantiles':quant(decoded[:,0]),'negative_turn_count':int((decoded[:,0]<0).sum()),'positive_turn_count':int((decoded[:,0]>0).sum())})
   archive[root.name+'_muscles']=muscles;archive[root.name+'_preactivation']=pre
  out=run/'readout-decomposition.npz';np.savez_compressed(out,**archive);result.update(status='completed_original_readout_decomposition',reports=reports,output_sha256=sha256(out),interpretation='Exact fixed readout algebra on previously examined fit-forbidden stationary and teacher-motion histories. No calibration, model modification, biological claim or driving claim.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_original_readout_decomposition':raise SystemExit(1)
if __name__=='__main__':main()
