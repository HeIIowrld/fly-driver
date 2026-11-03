"""Bounded full-history finite-difference tuning, synthetic only, no actor."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch,yaml
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flyvis_parameter_integrity import assert_nonnegative_synapse_strength,project_nonnegative_bounded
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=read(a.plan);root=Path(p['model_dir']);cp=Path(p['initial_candidate']);flash=Path(p['flash_run']);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),actor=False,seed=p['seed']));out=dict(status='failed',actor_deployed=False,vehicle_control_sent=False,optimizer_updates=0)
 try:
  if sha256(root/'best_chkpt')!=p['checkpoint_sha256'] or sha256(cp)!=p['initial_candidate_sha256'] or sha256(flash/'responses.npz')!=p['flash_sha256'] or sha256(flash/'stimulus.json')!=p['stimulus_sha256']:raise ValueError('pinned inputs differ')
  cache=Path(p['cache']);os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());install_datamate_cache_compat(cache);import flyvis
  torch.set_num_threads(1);torch.manual_seed(p['seed']);torch.set_default_device('cuda')
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  config=yaml.safe_load((root/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];net=flyvis.Network(**config).to('cuda');original=torch.load(root/'best_chkpt',map_location='cpu',weights_only=True)['network'];net.load_state_dict(original,strict=True);net.eval().requires_grad_(False);weight=net.edges_syn_strength;origin=weight.detach().clone();assert_nonnegative_synapse_strength(origin)
  keys=net.edge_params.syn_strength.keys;indices=[i for i,k in enumerate(keys) if isinstance(k,tuple) and len(k)==2 and k[1]=='T4d']
  if [list(keys[i]) for i in indices]!=p['allowed_connections']:raise ValueError('incoming topology mismatch')
  mask=torch.zeros_like(weight,dtype=torch.bool);mask[indices]=True;other={name:v.detach().clone() for name,v in net.named_parameters() if name!='edges_syn_strength'}
  with np.load(flash/'responses.npz',allow_pickle=False) as z:x=torch.as_tensor(z['stimulus'][[0,2],:401].copy(),dtype=torch.float32,device='cuda')
  par=read(flash/'stimulus.json')['parameters']
  if [(par[i]['radius'],par[i]['intensity']) for i in (0,2)]!=[(-1,0),(-1,1)] or tuple(x.shape)!=(2,401,1,721):raise ValueError('declared full-field flashes required')
  central=net.connectome.central_cells_index[:];names=net.connectome.nodes.type[:].astype(str)[central];cells=['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'];idx=central[[int(np.flatnonzero(names==c)[0]) for c in cells]];times=np.arange(401)*.005-1.;window=(times>=-.005)&(times<=1.);window_tensor=torch.as_tensor(window,device='cuda')
  evaluations=[]
  def evaluate(weights,label):
   with torch.no_grad():
    projected=project_nonnegative_bounded(weights,origin,mask,p['max_abs_weight_change']);weight.copy_(projected);response=net.simulate(x,dt=.005,initial_state='auto')[:,:,idx][:,window_tensor,:];offset=response.amin(dim=(0,1)).abs();adjust=response+offset;peaks=adjust.amax(dim=1);values=((peaks[1]-peaks[0])/(peaks[1]+peaks[0]+1e-16)).detach().cpu().numpy();torch.cuda.synchronize()
   if not np.isfinite(values).all():raise ValueError('nonfinite flash index')
   evaluations.append(dict(label=label,indices=values.tolist()));write_json(run/'progress.json',dict(evaluations=len(evaluations),planned=25,last=label,t4d=float(values[3])));return values
  ref=evaluate(origin,'original')
  if abs(float(ref[3])-p['original_t4d_index'])>2e-5:raise ValueError('author baseline index mismatch')
  payload=torch.load(cp,map_location='cpu',weights_only=True)
  if payload['actor_qualified'] is not False or payload['source_checkpoint_sha256']!=p['checkpoint_sha256']:raise ValueError('initial candidate lineage')
  proposed=payload['network']['edges_syn_strength'].to('cuda');assert_nonnegative_synapse_strength(proposed)
  if not torch.equal(proposed[~mask],origin[~mask]):raise ValueError('non-target candidate changed')
  candidate=project_nonnegative_bounded(proposed,origin,mask,p['max_abs_weight_change']).detach().clone();initial=evaluate(candidate,'initial')
  def objective(v):return float(max(0.,p['target_index']-v[3])**2+p['other_cell_penalty']*np.mean((v[[0,1,2,4,5,6,7]]-ref[[0,1,2,4,5,6,7]])**2))
  initial_loss=objective(initial);gradient=[];eps=p['finite_difference_step']
  for index in indices:
   high=candidate.clone();low=candidate.clone();high[index]+=eps;low[index]-=eps;high=project_nonnegative_bounded(high,origin,mask,p['max_abs_weight_change']);low=project_nonnegative_bounded(low,origin,mask,p['max_abs_weight_change']);span=float((high[index]-low[index]).cpu())
   if span<=0:raise ValueError('no finite difference span')
   gradient.append((objective(evaluate(high,f'{index}_high'))-objective(evaluate(low,f'{index}_low')))/span)
  g=np.asarray(gradient);maximum=float(np.max(np.abs(g)))
  if not np.isfinite(g).all() or maximum<1e-12:raise ValueError('no finite full-history gradient')
  direction=g/maximum;choices=[];best_loss=initial_loss;best=candidate;best_idx=initial;best_step=0.
  for step in p['line_search_steps']:
   trial=candidate.clone()
   with torch.no_grad():trial[indices]-=torch.as_tensor(direction*step,dtype=trial.dtype,device='cuda')
   trial=project_nonnegative_bounded(trial,origin,mask,p['max_abs_weight_change']);v=evaluate(trial,f'step_{step}');loss=objective(v);choices.append(dict(step=step,objective=loss,t4d_index=float(v[3])))
   if loss<best_loss:best_loss=loss;best=trial.detach().clone();best_idx=v;best_step=step
  with torch.no_grad():weight.copy_(best)
  assert_nonnegative_synapse_strength(weight)
  if not torch.equal(weight[~mask],origin[~mask]):raise ValueError('non-target change')
  for name,value in net.named_parameters():
   if name!='edges_syn_strength' and not torch.equal(value,other[name]):raise ValueError('non-target parameter changed')
  dest=run/'candidate.pt';torch.save(dict(network={k:v.detach().cpu() for k,v in net.state_dict().items()},actor_qualified=False,training_admissible=False,source_checkpoint_sha256=p['checkpoint_sha256'],plan_sha256=sha256(a.plan)),dest)
  out.update(status='completed_t4d_fullhistory_fd',optimizer_updates=int(best_step>0),evaluations=len(evaluations),initial_t4d_index=float(initial[3]),candidate_t4d_index=float(best_idx[3]),original_t4d_index=float(ref[3]),initial_objective=initial_loss,candidate_objective=best_loss,selected_step=best_step,gradient=gradient,choices=choices,evaluation_rows=evaluations,max_weight_change=float((weight-origin).abs().max().cpu()),checkpoint_sha256=sha256(dest),actor_qualified=False,training_admissible=False,physical_or_physiology_verified=False,interpretation='Synthetic full-field author-index objective. Bounded nonnegative 10 T4d incoming strengths; finite difference and fixed line search. Radius6 and direction remain held out; no actor or physiological qualification.')
 except Exception as e:out.update(error=type(e).__name__,reason=str(e),evaluations=len(locals().get('evaluations',[])))
 finally:write_json(run/'result.json',out)
 print(json.dumps({k:v for k,v in out.items() if k!='evaluation_rows'},allow_nan=False))
 if out['status']!='completed_t4d_fullhistory_fd':raise SystemExit(1)
if __name__=='__main__':main()
