from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive.env import FlyBrainEnv
from flybrain_drive.load_transfer_audit import diagnostic_snapshot, diagnostic_step


@pytest.mark.parametrize("wheel_is_geom2", [True, False])
def test_contact_frame_and_geom_order_keep_ground_on_wheel_force_and_moment(wheel_is_geom2):
    frame = np.eye(3) if wheel_is_geom2 else np.diag([-1., 1., -1.])
    force, torque = np.array([20., 3., 4.]), np.array([5., 6., 7.])
    sign = 1 if wheel_is_geom2 else -1
    wrench = np.r_[sign * (frame @ force), sign * (frame @ torque)]
    positions = np.array([.2, .3, .1])
    contact = SimpleNamespace(geom1=0 if wheel_is_geom2 else 1,
                              geom2=1 if wheel_is_geom2 else 0, frame=frame.ravel(), pos=positions)
    ids = {"terrain": 0, "wheel_fl": 1, "wheel_fr": 2, "wheel_rl": 3, "wheel_rr": 4}
    model = SimpleNamespace(body=lambda _: SimpleNamespace(id=1),
        geom=lambda name: SimpleNamespace(id=ids[name]), body_subtreemass=np.array([0, 10]))
    data = SimpleNamespace(subtree_com=np.zeros((2, 3)), subtree_linvel=np.zeros((2, 3)),
        subtree_angmom=np.zeros((2, 3)), time=0, ncon=1, contact=[contact])

    def contact_force(model, data, index, result):
        result[:] = wrench

    env = SimpleNamespace(model=model, data=data,
        mj=SimpleNamespace(mj_subtreeVel=lambda *_: None, mj_contactForce=contact_force),
        vehicle=SimpleNamespace(suspension_stiffness=10, suspension_damping=1),
        _joint_positions=lambda _: np.zeros(4), _joint_velocities=lambda _: np.zeros(4),
        susp_joints=[])
    result = diagnostic_snapshot(env)
    np.testing.assert_allclose(result["ground_force_world_N"][0], force)
    np.testing.assert_allclose(result["ground_moment_about_com_world_Nm"],
                               np.cross(positions, force) + torque)
    np.testing.assert_allclose(result["contact_normal_N"], [20, 0, 0, 0])
    assert "actor" not in result


def test_native_mujoco_static_contact_weight_and_momentum_balance():
    env = FlyBrainEnv(mode="O", actuation_mode="off", road={
        "flattened": True, "turn_amplitude": 0, "width": 36})
    try:
        env.reset(seed=770901, options={"start_x": 20})
        samples = [diagnostic_step(env, [-1, 0, 0]) for _ in range(120)]
        selected = samples[-20:]
        weight = selected[0]["mass_kg"] * 9.81
        force = np.mean([s["mean_ground_force_world_N"] for s in selected], axis=0)
        assert abs(force[:, 2].sum() / weight - 1) < .01
        assert np.min(force[:, 2]) > 0
        assert np.linalg.norm(force.sum(0)[:2]) / weight < .001
        assert max(np.linalg.norm(s["linear_balance_residual_N"]) for s in selected) / weight < .02
        assert all(s["unexpected_contacts_during_step"] == 0 for s in selected)
    finally:
        env.close()
