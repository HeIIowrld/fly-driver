"""TRAIN-only recalibration of the existing engineering DN visual-intent projection."""
import torch
from scripts.functional_priority_experiment import Experiment
from flybrain_drive.functional_motion_overlay import exact_model

class DescendingExperiment(Experiment):
    def __init__(self, plan):
        super().__init__(plan)
        self.gain.requires_grad_(False)
        self.dn.projection.requires_grad_(True)
        self.initial_weight=self.dn.projection.weight.detach().clone()
    def verify_fixed(self):
        for index,(module,old) in enumerate(zip([*self.components[:5],self.motion],self.protected)):
            state=module.state_dict()
            if index==2:
                state={k:v for k,v in state.items() if k!='projection.weight'}
                old={k:v for k,v in old.items() if k!='projection.weight'}
            if not exact_model(state,old):raise ValueError('Protected component changed '+str(index))
        if not torch.equal(self.gain.value,torch.ones_like(self.gain.value)):raise ValueError('Temporal gain changed')
        if float(abs(self.dn.projection.weight-self.initial_weight).max())>self.plan['maximum_weight_delta']+1e-7:raise ValueError('Projection bound exceeded')
    @torch.no_grad()
    def project(self):
        self.dn.projection.weight.copy_(self.initial_weight+(self.dn.projection.weight-self.initial_weight).clamp(-self.plan['maximum_weight_delta'],self.plan['maximum_weight_delta']))
