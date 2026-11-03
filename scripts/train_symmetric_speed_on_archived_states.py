"""Train bounded symmetric speed adapter on previously stored synthetic state only."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flight_state_speed_command import SymmetricFlightStateSpeedCommand
from flybrain_drive.graph import sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 root=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-symmetry-pilot-1');r=json.loads((root/'result.json').read_text());f=root/'multispeed-pilot.npz'
 if r['status']!='completed_synthetic_multispeed_symmetry_pilot' or sha256(f)!=r['output_sha256']:raise ValueError('Exact synthetic state source required')
 with np.load(f,allow_pickle=False) as z:
  design=z['fit_design'].copy();cent=z['center_coeff'].copy();target=z['fit_target'].copy();evaluation={str(v):z[f'{v:g}_evaluation_muscles'].copy() for v in (1.5,3.,4.5)};off={str(v):z[f'{v:g}_off_muscles'].copy() for v in (1.5,3.,4.5)}
 if design.shape!=(90,10) or cent.shape!=(2,6) or target.shape!=(90,):raise ValueError('Exact archived synthetic feature layout required')
 speeds=np.repeat([1.5,3.,4.5],30);muscles=design[:,:6]*1e-6+cent[0]+speeds[:,None]*cent[1]
 if not np.isfinite(muscles).all():raise ValueError('Finite original synthetic muscles required')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),source_sha256=sha256(f),source_result_sha256=sha256(root/'result.json'),seed=863009,training_seed=881403,evaluation_seed=881511,previously_examined_evaluation=True,optimizer='Adam lr=.02, 400 updates',target='hypothetical straight 3.5 and bend 2.5 m/s, not actual track speed',actor=False,fit_real=False))
 result=dict(status='failed',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False)
 try:
  torch.manual_seed(863009);net=SymmetricFlightStateSpeedCommand(6,max_delta_m_s=5.,pair_scale=.001,max_gain=2.);optim=torch.optim.Adam(net.parameters(),lr=.02)
  m=torch.tensor(muscles,dtype=torch.float32);sp=torch.tensor(speeds[:,None],dtype=torch.float32);goals=torch.tensor([[0.,1.]]).expand(len(m),-1);y=torch.tensor(target[:,None],dtype=torch.float32);losses=[]
  for i in range(400):
   optim.zero_grad();out=net(m,sp,goals)[:,1:2];loss=(out-y).square().mean()
   if not torch.isfinite(loss):raise ValueError('finite training loss required')
   loss.backward();optim.step();losses.append(float(loss.detach()));result['optimizer_updates']=i+1
  def evaluate(x,v):
   x=torch.tensor(x,dtype=torch.float32);speed=torch.full((len(x),1),v);g=torch.tensor([[0.,1.]]).expand(len(x),-1)
   with torch.no_grad():pred=net(x,speed,g)[:,1].numpy();plus=x.clone();plus[:,0]+=1e-6;perturb=(net(plus,speed,g)[:,1]-torch.from_numpy(pred)).abs().numpy()
   means=pred.reshape(3,10).mean(1);return dict(means=means.tolist(),straight_minus_bend=float(means[0]-means[1:].mean()),mse=float(np.mean((pred-np.repeat([3.5,2.5,2.5],10))**2)),max_1e_6_muscle_perturbation_m_s=float(perturb.max()))
  findings=[dict(speed_m_s=v,evaluation=evaluate(evaluation[str(v)],v),spatial_off=evaluate(off[str(v)],v)) for v in (1.5,3.,4.5)]
  dest=run/'symmetric-speed-candidate.pt';torch.save(dict(schema='flybrain.synthetic.symmetric-speed-boundary.v1',model=net.state_dict(),max_delta_m_s=5.,pair_scale=.001,max_gain=2.,actor_qualified=False,synthetic_only=True),dest)
  result.update(status='completed_synthetic_symmetric_speed_state_fit',training_loss_initial=losses[0],training_loss_final=losses[-1],conditions=findings,candidate_sha256=sha256(dest),optimizer_updates=400,fit_real=False,evaluation_previously_examined=True,closed_loop_verified=False,interpretation='Synthetic fixed archived muscle states, one train and one evaluation texture; no recurrent rerun, physics or driving proof')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result))
 if result['status']!='completed_synthetic_symmetric_speed_state_fit':raise SystemExit(1)
if __name__=='__main__':main()
