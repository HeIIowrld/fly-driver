import pytest
from flybrain_drive.first_gear_freeze import require_first_gear_freeze_state


def state():return dict(paused=False,live=True,in_menu=False,is_online=False,ai_controlled=False,camera_active=False,expert_probe_active=False)
def fields():return dict(gear=2,autoShifterOn=1,speedKmh=0.,gas=0.,brake=1.)


def test_observed_first_gear_brake_hold_can_be_frozen():require_first_gear_freeze_state(state(),fields())


@pytest.mark.parametrize('key,value',[('gear',1),('autoShifterOn',0),('speedKmh',.1),('gas',.02),('brake',.5)])
def test_first_gear_freeze_does_not_admit_neutral_motion_or_unheld_brake(key,value):
    row=fields();row[key]=value
    with pytest.raises(ValueError):require_first_gear_freeze_state(state(),row)
