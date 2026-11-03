"""Target full-window flash peaks using bounded truncated gradients; no actor."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch,yaml
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.graph import sha256

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();plan=read(a.plan);root=Path(plan['model_dir']);flash=Path(plan['flash_run']);run=begin_run(a.output,dict(plan_sha256=sha256(a.plan),program_sha256=sha256(__file__),original_sha256=sha256(root/'best_chkpt'),actor=False,seed=plan['seed']));result=dict(status='failed',optimizer_updates=0,actor_deployed=False,vehicle_control_sent=False)
 try:
  cp=Path(plan['initial_candidate'])
  if sha256(root/'best_chkpt')!=plan['checkpoint_sha256'] or sha256(cp)!=plan['initial_candidate_sha256'] or sha256(flash/'responses.npz')!=plan['flash_sha256'] or sha256(flash/'stimulus.json')!=plan['stimulus_sha256']:raise ValueError('exact pinned inputs required')
  cache=Path(plan['cache']);os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());install_datamate_cache_compat(cache);import flyvis
  torch.set_num_threads(1);torch.manual_seed(plan['seed']);torch.set_default_device('cuda')
  if not torch.cuda.is_available():raise RuntimeError('GPU required')
  cfg=yaml.safe_load((root/'_meta.yaml').read_text(encoding='utf8'))['config']['network'];net=flyvis.Network(**cfg).to('cuda');original=torch.load(root/'best_chkpt',map_location='cpu',weights_only=True)['network'];net.load_state_dict(original,strict=True);net.eval().requires_grad_(False)
  keys=net.edge_params.syn_strength.keys;indices=[i for i,k in enumerate(keys) if isinstance(k,tuple) and len(k)==2 and k[1]=='T4d']
  if [list(keys[i]) for i in indices]!=plan['allowed_connections']:raise ValueError('incoming connection mismatch')
  weight=net.edges_syn_strength;original_weight=weight.detach().clone();others={name:p.detach().clone() for name,p in net.named_parameters() if name!='edges_syn_strength'};allowed=torch.zeros_like(weight,dtype=torch.bool);allowed[indices]=True
  with np.load(flash/'responses.npz',allow_pickle=False) as z:movies=z['stimulus'][[0,2]].copy()
  params=read(flash/'stimulus.json')['parameters']
  if [(params[i]['radius'],params[i]['intensity']) for i in (0,2)]!=[(-1,0),(-1,1)] or movies.shape!=(2,600,1,721):raise ValueError('full-field flashes required')
  x=torch.as_tensor(movies,dtype=torch.float32,device='cuda');central=net.connectome.central_cells_index[:];names=net.connectome.nodes.type[:].astype(str)[central];cells=['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'];idx=central[[int(np.flatnonzero(names==c)[0]) for c in cells]]
  def prepare():
   with torch.no_grad():
    st=net.steady_state(1.,.005,2);net.stimulus.zero(2,200);net.stimulus.add_input(x[:,:200]);return net(net.stimulus(),.005,st,as_states=True)[-1]
  def forward(start,end,state,states=False):
   net.stimulus.zero(2,end-start);net.stimulus.add_input(x[:,200+start:200+end]);return net(net.stimulus(),.005,state,as_states=states)
  with torch.no_grad():ref=forward(0,201,prepare())[:,:,idx].detach()
  initial=torch.load(cp,map_location='cpu',weights_only=True)
  if initial['actor_qualified'] is not False or initial['source_checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('initial candidate lineage')
  net.load_state_dict(initial['network'],strict=True);weight.requires_grad_(True);opt=torch.optim.Adam([weight],lr=plan['learning_rate']);losses=[];peak_frames=[];updates=0
  for step in range(plan['updates']):
   with torch.no_grad():
    warm=prepare();preview=forward(0,201,warm)[:,:,idx].detach();peak=preview[:,:,3].argmax(dim=1).cpu().tolist();start=max(0,min(peak)-plan['history_frames']+1);end=max(peak)+1
    if end-start>plan['max_gradient_frames']:raise ValueError('peak separation exceeds memory bound')
    segment_state=warm if start==0 else forward(0,start,warm,states=True)[-1]
   opt.zero_grad(set_to_none=True);value=forward(start,end,segment_state)[:,:,idx];contrast=value[1,peak[1]-start,3]-value[0,peak[0]-start,3];preserve=(value[:,:,[0,1,2,4,5,6,7]]-ref[:,start:end,[0,1,2,4,5,6,7]]).square().mean();loss=torch.relu(plan['target_peak_difference']-contrast).square()+plan['preservation_penalty']*preserve
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward()
   if weight.grad is None or not torch.isfinite(weight.grad).all():raise ValueError('nonfinite gradient')
   with torch.no_grad():weight.grad[~allowed]=0
   opt.step()
   with torch.no_grad():weight[allowed]=torch.maximum(torch.minimum(weight[allowed],original_weight[allowed]+plan['max_abs_weight_change']),original_weight[allowed]-plan['max_abs_weight_change']);weight[~allowed]=original_weight[~allowed]
   updates+=1;losses.append(float(loss.detach().cpu()));peak_frames.append(peak);write_json(run/'progress.json',dict(updates=updates,loss=losses[-1],peak_indices=peak,peak_difference=float(contrast.detach().cpu())))
  weight.requires_grad_(False)
  with torch.no_grad():final=forward(0,201,prepare())[:,:,idx].detach()
  for name,p in net.named_parameters():
   if name!='edges_syn_strength' and not torch.equal(p,others[name]):raise ValueError('non-target parameter changed')
  if not torch.equal(weight[~allowed],original_weight[~allowed]):raise ValueError('non-target edge changed')
  dest=run/'candidate.pt';torch.save(dict(network={k:v.detach().cpu() for k,v in net.state_dict().items()},actor_qualified=False,training_admissible=False,source_checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=sha256(a.plan)),dest);data=run/'window.npz';np.savez_compressed(data,reference=ref.cpu().numpy(),candidate=final.cpu().numpy(),cells=np.array(cells));result.update(status='completed_t4d_synthetic_peak_fit',optimizer_updates=updates,losses=losses,peak_indices=peak_frames,original_peak_difference=float((ref[1,:,3].max()-ref[0,:,3].max()).cpu()),candidate_peak_difference=float((final[1,:,3].max()-final[0,:,3].max()).cpu()),max_weight_change=float((weight-original_weight).abs().max().cpu()),checkpoint_sha256=sha256(dest),output_sha256=sha256(data),actor_qualified=False,training_admissible=False,physical_or_physiology_verified=False,interpretation='Full 201-frame peak selection, truncated local gradient; only full-field flashes fit; radius6 and moving-edge remain held out. No actor or physiological qualification.')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e),optimizer_updates=locals().get('updates',0))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']!='completed_t4d_synthetic_peak_fit':raise SystemExit(1)
if __name__=='__main__':main()
