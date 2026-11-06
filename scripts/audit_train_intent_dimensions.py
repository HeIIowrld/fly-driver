"""TRAIN-only command dimensions; not whole-vision rank or physiology audit."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.descending_intent_experiment import DescendingExperiment

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1);e=DescendingExperiment(read(a.plan));out=begin_run(a.output,dict(program_sha256=sha256(__file__),plan_sha256=sha256(a.plan),parent_candidate=e.parent,scope=__doc__,optimizer_updates=0,vehicle_control_sent=False));commands=[];power=[];speeds=[]
 def capture(module,args,value):
  commands.append(args[0].detach().numpy().copy());speeds.append(args[1][:,16:17].detach().numpy().copy());power.append(value[1].detach().numpy().copy())
 hook=e.model.motor.command.register_forward_hook(capture)
 try:
  e.model.reset_episode();e.holder.update(mode='real',episode=0)
  for i in range(64):e.step(i,False)
  feature=e.extras[0][256:257].detach().clone().requires_grad_(True);jac=torch.autograd.functional.jacobian(lambda v:e.dn(v)[0],feature)[:,0,:].detach().numpy();state=e.dn.state_dict();normalized=((feature.detach()-state['reference'])/state['scale']).clamp(-5,5);raw=float((normalized@state['projection.weight'].T)[0,0]);contrast=8*np.tanh(raw);logits=state['tonic_logit'].numpy()+contrast*state['orientations'].numpy();sigmoid=1/(1+np.exp(-logits));v=.2*sigmoid*(1-sigmoid)*state['orientations'].numpy()*8*(1-np.tanh(raw)**2);unclipped=(abs(((feature.detach()-state['reference'])/state['scale']).numpy()[0])<5);expected=np.outer(v,state['projection.weight'].numpy()[0]/state['scale'].numpy()*unclipped);error=float(abs(jac-expected).max());singular=np.linalg.svd(expected.astype(float),compute_uv=False);relative_rank=int(np.sum(singular>max(singular[0]*1e-6,1e-15)));assert relative_rank<=1 and error<2e-6
  command=np.concatenate(commands);speed=np.concatenate(speeds);power=np.concatenate(power);difference=float(abs(command[:,1:2]-speed).max());assert difference==0.;np.savez_compressed(out/'observations.npz',commands=command,speed=speed,power=power,jacobian=jac);e.verify_fixed();result=dict(status='completed_train_intent_dimension_audit',actual_train_events=64,desired_speed_minus_body_speed_maximum=difference,power_current_range=[float(power.min()),float(power.max())],power_ports=power.shape[1],recurrent_speed_boundary_attached=e.model.speed_command_boundary is not None,existing_visual_DN_projection_shape=list(state['projection.weight'].shape),four_DN_input_jacobian_singular_values=singular.tolist(),rank_relative_1e6=relative_rank,analytic_autograd_maximum_difference=error,observations_sha256=sha256(out/'observations.npz'),scope='Only existing engineering static-vision DN command projection and neutral speed-error input. Other static/motion/body paths still feed recurrent flight graph and may affect all muscles/pedals. Not total actor visual dimension, physiological control rank or proof of cause of poor steering.',optimizer_updates=0,vehicle_control_sent=False,actor_qualified=False);write_json(out/'result.json',result);print(json.dumps(result))
 finally:hook.remove();e.close()
if __name__=='__main__':main()
