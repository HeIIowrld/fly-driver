"""Shared fixed parent sensory experiment; no simulator/actuator interfaces."""
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_motion_spatial_candidate import load_readonly_motion_candidate
from flybrain_drive.functional_priority_objective import gradient_snapshot,gradient_restore,split_functional_loss,recording_losses
from flybrain_drive.functional_motion_overlay import exact_model
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.racing_goal import racing_goal
from flybrain_drive.sac_unified_student import read
from flybrain_drive.temporal_port_gain import TemporalPortGain,attach_temporal_gain
from flybrain_drive.temporal_body_context import per_view_temporal_body_context
from flybrain_drive.virtual_navigation_body import NavigationBodyObservation
from scripts.balanced_visual_sources import load_balanced_sources
from scripts.evaluate_controlled_motion_factorial import synthetic_body
from scripts.motion_spatial_experiment import load_motion_experiment
from scripts.sac_steering_experiment import snapshot,restore
from scripts.train_qualified_sac_unified import context

class Experiment:
    def __init__(self,plan):
        self.plan=plan
        for path,value in plan['source_sha256'].items():
            if sha256(path)!=value:raise ValueError('Sealed experiment source changed '+path)
        if sha256(plan['candidate_plan'])!=plan['candidate_plan_sha256']:raise ValueError('Parent plan changed')
        self.components,self.motion,self.parent=load_readonly_motion_candidate(read(plan['candidate_plan']));self.model,self.head,self.dn,self.optical,self.body_port,_=self.components;old=Path(read(plan['candidate_plan'])['motion']['fit']);om=read(old/'manifest.json')['config'];_,_,self.entries,self.items,self.extras,self.native,self.valid,_,self.pre,self.features,self.maps,_,_,_=load_motion_experiment(om['plan'],'fine_motion')
        with np.load(old/'predictions.npz') as z:self.anchors={e:z[f'prediction_{e}'].copy() for e in (0,1)}
        self.sources=load_balanced_sources(plan,self.parent)
        for module in [*self.components[:5],self.motion]:module.eval().requires_grad_(False)
        self.protected=[{k:v.clone() for k,v in module.state_dict().items()} for module in [*self.components[:5],self.motion]];self.gain=TemporalPortGain(len(self.model.motor.backbone.vision_ports),plan['maximum_gain_change']);self.holder={};self.goals=torch.tensor(racing_goal(300,1));self.views=torch.ones(1,4,dtype=torch.bool);self.contract=sha256('configs/ac-virtual-navigation-body-v2.json');self.real_tasks=[]
        for e in (0,1):
            d=self.items[e][0];blocks=(len(d['scene'])+plan['window']-1)//plan['window'];selected=np.union1d(np.unique(np.flatnonzero(self.pre[e])//plan['window']),np.linspace(0,blocks-1,plan['anchor_blocks_per_episode'],dtype=int));self.real_tasks.extend(dict(episode=e,begin=int(b*plan['window']),end=min(int((b+1)*plan['window']),len(d['scene']))) for b in selected)
        prefix=Path(plan['prefix_run']);pr=read(prefix/'result.json');pm=read(prefix/'manifest.json')['config'];self.prefix_file=prefix/'parent-prefix-states.pt'
        if sha256(prefix/'result.json')!=plan['prefix_run_result_sha256'] or sha256(self.prefix_file)!=pr['parent_prefix_states_sha256'] or pm['parent_candidate']!=self.parent or pr['parent_full_train_max_differences']!=[0.,0.] or {(v['episode'],v['begin'],v['end']) for v in self.real_tasks}!={(v['episode'],v['begin'],v['end']) for v in pm['real_windows']}:raise ValueError('Exact audited parent prefix/window binding required')
        self.states=torch.load(self.prefix_file,map_location='cpu',weights_only=True);self.hooks=[self.model.motor.command.register_forward_hook(self._descending),self.model.visual.register_forward_hook(self._visual),self.model.body_feedback.register_forward_hook(self._body),attach_temporal_gain(self.model.temporal,self.gain)];torch.set_flush_denormal(True);self.backend=prepare_fixed_cpu_backend(self.model)
    def inputs(self):
        h=self.holder;i=h['index']
        if h['mode']=='real':
            e=h['episode'];j=int(self.maps[e][i]);return self.extras[e][i:i+1],self.features[e][j:j+1],self.native[e][i:i+1],self.valid[e][i:i+1]
        x=h['source']['x'];k=h['case'];return torch.tensor(x['static'][k,i:i+1]),torch.tensor(x['motion'][k,i:i+1]),h['native'],h['native_valid']
    def _descending(self,module,args,value):
        static,_,_,_=self.inputs();steer,power=value;return steer+self.dn(static),power
    def _visual(self,module,args,value):
        static,fine,_,_=self.inputs();return value+self.optical(static[:,64:])+self.motion(fine)
    def _body(self,module,args,value):
        _,_,native,valid=self.inputs();return value+self.body_port(native,valid)
    def step(self,i,learn):
        h=self.holder;h['index']=i
        if h['mode']=='real':
            d,c,_=self.items[h['episode']];scene=torch.tensor(d['scene'][i:i+1]);ctx=context(d,c,i);now=float(d['decision_time'][i])
        else:
            x=h['source']['x'];clocks=tuple(map(float,x['render_lowers'][i]));camera=float(x['camera_receipt'][i]);now=camera+.005;values,mask=h['body'],h['mask'];current=NavigationBodyObservation('synthetic-factorial',1000+i,now-.001,now,values.copy(),mask.copy(),self.contract);history=tuple(NavigationBodyObservation('synthetic-factorial',10*i+j,t-.001,camera,values.copy(),mask.copy(),self.contract) for j,t in enumerate(clocks));ctx=per_view_temporal_body_context(history,current,render_start_lowers=clocks,camera_receipt_time=camera,decision_time=now);scene=torch.tensor(x['scene'][h['case'],i:i+1])
        with torch.set_grad_enabled(learn):
            fn=self.model.sensory_training_decision if learn else self.model.sensory_decision;_,muscles=fn(decision_time=now,racing_goals=self.goals,scene_features=scene,scene_valid=self.views,body_context=ctx);return self.head(muscles)[0]
    def real(self,task):
        e=task['episode'];begin,end=task['begin'],task['end'];self.holder.update(mode='real',episode=e);restore(self.model,self.states[e,begin]);d=self.items[e][0];prediction=torch.stack([self.step(i,True) for i in range(begin,end)]);anchor=torch.tensor(self.anchors[e][begin:end]);target=torch.tensor(d['labels'][begin:end]);fit=torch.tensor(d['fit_mask'][begin:end]);pre=torch.tensor(self.pre[e][begin:end]);losses=recording_losses(prediction,anchor,target,fit,pre,anchor_weight=self.plan['real_anchor_weight'],maximum_ratio=self.plan['actual_recording_preservation_maximum_ratio'],protection_weight=self.plan.get('protection_weight',1.));return losses,prediction
    def synthetic(self,task,mode):
        if mode not in ('legacy','branch_full','episode_full'):raise ValueError('Explicit gradient regime required')
        source=self.sources[task['source_slot']];x,sm,stimulus=source['x'],source['metadata'],source['plan'];k=task['pair'];case=sm['cases'][k];body,mask,native,valid=synthetic_body(case['speed_mps'],task['temperature'],self.body_port.reference.numpy());self.holder.update(mode='synthetic',source=source,case=k,body=body,mask=mask,native=torch.tensor(native[None],dtype=torch.float32),native_valid=torch.tensor(valid[None]));self.model.reset_episode()
        for i in range(stimulus['branch_frame']):self.step(i,mode=='episode_full')
        prefix=gradient_snapshot(self.model) if mode=='episode_full' else snapshot(self.model);branches=[]
        for case_id in (k,k+1):
            self.holder['case']=case_id;(gradient_restore if mode=='episode_full' else restore)(self.model,prefix);sequence=[]
            for i in range(stimulus['branch_frame'],stimulus['frame_count']):
                value=self.step(i,mode!='legacy' or i>=stimulus['evaluation_first_frame'])
                if i>=stimulus['evaluation_first_frame']:sequence.append(value)
            branches.append(torch.stack(sequence))
        anchor=torch.tensor(source['anchor'][f"group_{task['group']}_fine_prediction"][:,stimulus['evaluation_first_frame']:]);losses=split_functional_loss(branches[0],branches[1],anchor,left_sign=stimulus['expected_canonical_steering_sign_per_left_bend'],target_margin=self.plan['target_margin'],direction_weight=self.plan['direction_weight'],anchor_weight=self.plan['synthetic_anchor_weight']);return losses,torch.stack(branches)
    def verify_fixed(self):
        if any(not exact_model(module.state_dict(),old) for module,old in zip([*self.components[:5],self.motion],self.protected)):raise ValueError('Protected original component changed')
    def close(self):
        for hook in self.hooks:hook.remove()
        torch.set_flush_denormal(False)
