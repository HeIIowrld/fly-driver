"""Experimental bounded T4d incoming-synapse fit on synthetic flashes only."""
import argparse,json,os,time
from pathlib import Path
import numpy as np,torch,yaml
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256
from flybrain_drive.flyvis_parameter_integrity import assert_nonnegative_synapse_strength, project_nonnegative_bounded

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=read(a.plan);root=Path(plan['model_dir']);flash=Path(plan['flash_run']);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),model_sha256=sha256(root/'best_chkpt'),actor=False,seed=plan['seed']));result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  if sha256(root/'best_chkpt')!=plan['checkpoint_sha256'] or sha256(flash/'responses.npz')!=plan['flash_sha256'] or sha256(flash/'stimulus.json')!=plan['stimulus_sha256']:raise ValueError('exact pinned model and stimulus required')
  cache=Path(plan['cache']);os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());install_datamate_cache_compat(cache);import flyvis
  torch.set_num_threads(1);torch.manual_seed(plan['seed']);torch.set_default_device('cuda')
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  cfg=yaml.safe_load((root/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];net=flyvis.Network(**cfg).to('cuda');original=torch.load(root/'best_chkpt',map_location='cpu',weights_only=True)['network'];net.load_state_dict(original,strict=True);net.eval().requires_grad_(False)
  keys=net.edge_params.syn_strength.keys;indices=[i for i,k in enumerate(keys) if isinstance(k,tuple) and len(k)==2 and k[1]=='T4d']
  if not indices or [list(keys[i]) for i in indices]!=plan['allowed_connections']:raise ValueError('exact incoming connections required')
  weight=net.edges_syn_strength;assert_nonnegative_synapse_strength(weight);start=weight.detach().clone();others={name:p.detach().clone() for name,p in net.named_parameters() if name!='edges_syn_strength'};allowed=torch.zeros_like(weight,dtype=torch.bool);allowed[indices]=True
  with np.load(flash/'responses.npz',allow_pickle=False) as z:movies=z['stimulus'][[0,2]].copy()
  params=read(flash/'stimulus.json')['parameters']
  if [(params[i]['radius'],params[i]['intensity']) for i in (0,2)]!=[(-1,0),(-1,1)] or movies.shape!=(2,600,1,721):raise ValueError('full-field training-only flashes required')
  x=torch.as_tensor(movies,dtype=torch.float32,device='cuda');central=net.connectome.central_cells_index[:];names=net.connectome.nodes.type[:].astype(str)[central];cells=['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'];indices_cell=[int(np.flatnonzero(names==c)[0]) for c in cells];idx=central[indices_cell]
  # Grey warm state is numerically continuous. Gradient is truncated at flash onset.
  with torch.no_grad():
   state=net.steady_state(1.,.005,2);net.stimulus.zero(2,200);net.stimulus.add_input(x[:,:200]);state=net(net.stimulus(),.005,state,as_states=True)[-1]
  def predict():
   net.stimulus.zero(2,plan['train_frames']);net.stimulus.add_input(x[:,200:200+plan['train_frames']]);return net(net.stimulus(),.005,state)[:,:,idx]
  with torch.no_grad():baseline=predict().detach()
  weight.requires_grad_(True);optimizer=torch.optim.Adam([weight],lr=plan['learning_rate']);losses=[];updates=0
  for j in range(plan['updates']):
   optimizer.zero_grad(set_to_none=True);y=predict();contrast=y[1,:,3].max()-y[0,:,3].max();preserve=(y[:,:,[0,1,2,4,5,6,7]]-baseline[:,:,[0,1,2,4,5,6,7]]).square().mean();loss=torch.relu(plan['target_contrast']-contrast).square()+plan['preservation_penalty']*preserve
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward()
   if weight.grad is None or not torch.isfinite(weight.grad).all():raise ValueError('nonfinite gradient')
   with torch.no_grad():weight.grad[~allowed]=0
   optimizer.step()
   with torch.no_grad():
    weight.copy_(project_nonnegative_bounded(weight,start,allowed,plan['max_abs_weight_change']))
   updates+=1;losses.append(float(loss.detach().cpu()));write_json(run/'progress.json',dict(updates=updates,total=plan['updates'],loss=losses[-1],contrast=float(contrast.detach().cpu())))
  weight.requires_grad_(False)
  with torch.no_grad():after=predict().detach()
  for name,p in net.named_parameters():
   if name!='edges_syn_strength' and not torch.equal(p,others[name]):raise ValueError('non-target parameter changed')
  if not torch.equal(weight[~allowed],start[~allowed]):raise ValueError('non-target edge changed')
  assert_nonnegative_synapse_strength(weight)
  dest=run/'candidate.pt';torch.save(dict(network={k:v.detach().cpu() for k,v in net.state_dict().items()},actor_qualified=False,training_admissible=False,source_checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha256(a.plan)),dest)
  data=run/'train_window.npz';np.savez_compressed(data,baseline=baseline.cpu().numpy(),candidate=after.cpu().numpy(),cells=np.array(cells));result.update(status='completed_t4d_nonnegative_fit',optimizer_updates=updates,losses=losses,incoming_connections=plan['allowed_connections'],baseline_contrast=float((baseline[1,:,3].max()-baseline[0,:,3].max()).cpu()),candidate_contrast=float((after[1,:,3].max()-after[0,:,3].max()).cpu()),max_weight_change=float((weight-start).abs().max().cpu()),checkpoint_sha256=sha256(dest),output_sha256=sha256(data),actor_qualified=False,training_admissible=False,physical_or_physiology_verified=False,interpretation='Full-field first 40 stimulus frames only; fixed no-gradient warm state, nonnegative strengths enforced and original graph unchanged. Must validate entire author histories and heldout stimuli.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),optimizer_updates=locals().get('updates',0))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_t4d_nonnegative_fit':raise SystemExit(1)
if __name__=='__main__':main()
