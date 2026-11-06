"""Optimizer layout and frozen full-routing contract, synthetic modules only."""
from types import SimpleNamespace
import pytest
import torch
from flybrain_drive.visual_dn_current_calibration import VisualDNCurrentCalibration
from scripts.full_range_port_experiment import RegisteredCurrent
from scripts.fixed_full_routing_experiment import FixedFullRoutingExperiment


def test_reject_nonfixed_plan_before_data_loading():
 with pytest.raises(ValueError):FixedFullRoutingExperiment(dict(initial_gate=.99,minimum_gate=1.),train_body=False)


def test_native_group_reindex_and_projection_preserve_fixed_gate():
 e=object.__new__(FixedFullRoutingExperiment);e.train_body=True;e.calibration=RegisteredCurrent(VisualDNCurrentCalibration(torch.zeros(320),torch.ones(320),torch.tensor([1.,-1.,1.,-1.])),1.);e.calibration.gate.requires_grad_(False);e.body_port=SimpleNamespace(projection=torch.nn.Linear(46,201,bias=False));e.current_origin=e.calibration.projection.weight.detach().clone();e.bias_origin=e.calibration.projection.bias.detach().clone();e.native_origin=e.body_port.projection.weight.detach().clone();e.port_plan=dict(visual_step_norm=.0015,visual_total_norm=2.,bias_step_norm=.002,bias_total_norm=2.,gate_step_norm=.01,minimum_gate=1.,native_step_norm=.00025,native_total_norm=.05,native_entry_delta=.02)
 parameters=e.train_parameters();assert len(parameters)==3 and e.native_parameter_index==2 and all(p is not e.calibration.gate for p in parameters)
 old=[p.detach().clone() for p in parameters]
 with torch.no_grad():
  for p in parameters:p.add_(.1)
 parameters[2].grad=torch.ones_like(parameters[2]);e.project_after_step(old)
 assert torch.equal(e.calibration.gate,torch.ones(4)) and not e.calibration.gate.requires_grad
 assert 0<float((parameters[2]-old[2]).norm())<=.0002501
 assert float((parameters[0]-old[0]).norm())<=.0015001


def test_runtime_wrapper_rejects_partial_gate():
 from flybrain_drive.fixed_full_routing_candidate import FixedFullRoutingCurrent
 with pytest.raises(ValueError):FixedFullRoutingCurrent(None,.99)
