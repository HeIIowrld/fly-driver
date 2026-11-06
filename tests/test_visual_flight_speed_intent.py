import pytest
import torch
from flybrain_drive.visual_flight_speed_intent import VisualFlightSpeedIntent
from flybrain_drive.flight_maneuver import ManeuverCurrents,MANEUVER_CONTRACT
from flybrain_drive.racing_goal import RACING_GOAL_CONTRACT

def fixture():
 m=VisualFlightSpeedIntent(torch.zeros(320),torch.ones(320),torch.zeros(19),torch.ones(19));args=[torch.ones(1,320),torch.tensor([[15.]]),torch.ones(1,1,dtype=torch.bool),torch.ones(1,19),torch.ones(1,19,dtype=torch.bool),torch.ones(1,4,dtype=torch.bool),torch.tensor([[0.,3.]])];return m,args

def test_neutral_exact_and_body_cannot_create_visual_intent():
 m,a=fixture()
 with torch.no_grad():m.body_gain.weight.fill_(.5)
 for native in (torch.ones(1,19)*100,torch.ones(1,19)*-100):
  a[3]=native;torch.testing.assert_close(m(*a),a[1],rtol=0,atol=0)
 with torch.no_grad():m.visual_projection.weight.fill_(.1)
 a[0].zero_();torch.testing.assert_close(m(*a),a[1],rtol=0,atol=0)

def test_positive_negative_speed_and_original_power_sign():
 m,a=fixture();currents=ManeuverCurrents([-1,1,-1,1],2);body=torch.zeros(1,21);body[:,16]=15.;valid=torch.ones(1,21,dtype=torch.bool);targets=[];powers=[]
 for weight in (-.01,0.,.01):
  with torch.no_grad():m.visual_projection.weight.fill_(weight)
  target=m(*a);targets.append(float(target));command=torch.cat([torch.zeros_like(target),target],1);_,power=currents(command,body,valid,a[-1],command_contract=MANEUVER_CONTRACT,goal_contract=RACING_GOAL_CONTRACT);powers.append(float(power[0,0]))
 assert targets[0]<targets[1]<targets[2] and powers[0]<powers[1]<powers[2]

def test_native_gain_cannot_reverse_sign_and_is_bounded():
 m,a=fixture()
 with torch.no_grad():m.visual_projection.weight.fill_(.001);m.body_gain.weight.fill_(1.)
 for value in (-100,100):
  a[3].fill_(value);target,delta,gain=m.evaluate(*a);assert target>a[1] and delta>0 and torch.exp(torch.tensor(-.5))-1e-6<=gain<=torch.exp(torch.tensor(.5))+1e-6

def test_invalid_native_values_ignored_and_missing_gain_is_neutral():
 m,a=fixture()
 with torch.no_grad():m.visual_projection.weight.fill_(.001);m.body_gain.weight.fill_(.4)
 a[4][:,3:]=False;baseline=m(*a).detach();a[3][:,3:]=100000;torch.testing.assert_close(m(*a),baseline,rtol=0,atol=0)
 a[4][:]=False;_,_,gain=m.evaluate(*a);torch.testing.assert_close(gain,torch.ones_like(gain),rtol=0,atol=0)

def test_target_clamps_to_zero_and_external_ceiling():
 m,a=fixture()
 for sign,speed,expected in [(-1,.1,0.),(1,83.,83.3333333333)]:
  with torch.no_grad():m.visual_projection.weight.fill_(sign*10)
  a[1].fill_(speed);torch.testing.assert_close(m(*a),torch.tensor([[expected]]),atol=1e-5,rtol=0)

def test_gradients_reach_retinal_and_native_gain_without_mutating_inputs():
 m,a=fixture()
 with torch.no_grad():m.visual_projection.weight.fill_(.0001)
 copies=[v.clone() for v in a];m(*a).sum().backward();assert m.visual_projection.weight.grad.abs().sum()>0 and m.body_gain.weight.grad.abs().sum()>0
 for before,after in zip(copies,a):torch.testing.assert_close(before,after,rtol=0,atol=0)

@pytest.mark.parametrize('fault',['camera','speed_missing','negative_speed','native_nan','mask_type','feature_shape','ceiling'])
def test_invalid_sensor_contract_rejected(fault):
 m,a=fixture()
 if fault=='camera':a[5][0,0]=False
 elif fault=='speed_missing':a[2][:]=False
 elif fault=='negative_speed':a[1][:]=-1
 elif fault=='native_nan':a[3][0,0]=float('nan')
 elif fault=='mask_type':a[4]=a[4].float()
 elif fault=='feature_shape':a[0]=torch.zeros(1,321)
 elif fault=='ceiling':a[-1][:,1]=4
 with pytest.raises(ValueError):m(*a)
