"""Solver contact/momentum diagnostics, deliberately outside ActorObservation."""

import numpy as np

WHEELS = ("fl", "fr", "rl", "rr")


def diagnostic_snapshot(env):
    """World-frame ground-on-wheel forces and moments about whole-vehicle COM.

    MuJoCo contact axes are rows; force acts from geom1 towards geom2.
    Contact torque is included, including torsional and rolling resistance.
    This diagnostic dictionary is not an actor observation. Selected load values
    may separately enter the declared instrumented-load sensor contract.
    """
    mj, model, data = env.mj, env.model, env.data
    mj.mj_subtreeVel(model, data)
    root = model.body("chassis").id
    com = data.subtree_com[root].copy()
    terrain = model.geom("terrain").id
    wheel_ids = {model.geom("wheel_" + name).id: index for index, name in enumerate(WHEELS)}
    forces, normal = np.zeros((4, 3)), np.zeros(4)
    moment = np.zeros(3)
    unexpected = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (contact.geom1, contact.geom2)
        if terrain not in pair:
            unexpected += 1
            continue
        wheel = pair[1] if pair[0] == terrain else pair[0]
        if wheel not in wheel_ids:
            unexpected += 1
            continue
        wrench = np.zeros(6)
        mj.mj_contactForce(model, data, index, wrench)
        sign = 1 if pair[1] == wheel else -1
        rotation = contact.frame.reshape(3, 3).T
        force, torque = sign * (rotation @ wrench[:3]), sign * (rotation @ wrench[3:])
        slot = wheel_ids[wheel]
        forces[slot] += force
        normal[slot] += wrench[0]
        moment += np.cross(contact.pos - com, force) + torque
    mass = float(model.body_subtreemass[root])
    return {"time": float(data.time), "mass_kg": mass, "com_world_m": com,
            "com_velocity_world_mps": data.subtree_linvel[root].copy(),
            "angular_momentum_about_com_world": data.subtree_angmom[root].copy(),
            "ground_force_world_N": forces, "contact_normal_N": normal,
            "ground_moment_about_com_world_Nm": moment,
            "unexpected_contacts": unexpected,
            "suspension_proxy_N": np.maximum(0.,
                env.vehicle.suspension_stiffness * env._joint_positions(env.susp_joints)
                + env.vehicle.suspension_damping * env._joint_velocities(env.susp_joints))}


def diagnostic_step(env, action):
    """No-render fixed-budget diagnostic step, not an environment/policy rollout.

    Integrate solver forces at each 1ms substep before evaluating momentum changes.
    Does not invoke route/reward/episode completion or create actor observations.
    Only the normal common-drive/steering/brake adapter issues actuator controls.
    """
    mj, model, data = env.mj, env.model, env.data
    omega = env._joint_velocities(env.spin_joints)
    env.data.ctrl[:], command = env.control.command(
        action, float(np.mean(omega[:2]) * env.vehicle.wheel_radius),
        float(data.sensor("gyro").data[2]), omega, env.dt)
    mj.mj_forward(model, data)
    before = diagnostic_snapshot(env)
    impulse, angular_impulse = np.zeros(3), np.zeros(3)
    force_integral, normal_integral, proxy_integral = np.zeros((4, 3)), np.zeros(4), np.zeros(4)
    unexpected = 0
    for _ in range(env.substeps):
        sample = diagnostic_snapshot(env)
        impulse += (sample["ground_force_world_N"].sum(0)
                    + sample["mass_kg"] * model.opt.gravity) * model.opt.timestep
        angular_impulse += sample["ground_moment_about_com_world_Nm"] * model.opt.timestep
        force_integral += sample["ground_force_world_N"] * model.opt.timestep
        normal_integral += sample["contact_normal_N"] * model.opt.timestep
        proxy_integral += sample["suspension_proxy_N"] * model.opt.timestep
        unexpected += sample["unexpected_contacts"]
        mj.mj_step(model, data)
        mj.mj_forward(model, data)
    after = diagnostic_snapshot(env)
    duration = after["time"] - before["time"]
    dv = after["com_velocity_world_mps"] - before["com_velocity_world_mps"]
    dl = (after["angular_momentum_about_com_world"]
          - before["angular_momentum_about_com_world"])
    rotation = data.xmat[model.body("chassis").id].reshape(3, 3)
    after.update(acceleration_body_mps2=rotation.T @ (dv / duration),
                 mean_ground_force_world_N=force_integral / duration,
                 mean_contact_normal_N=normal_integral / duration,
                 mean_suspension_proxy_N=proxy_integral / duration,
                 linear_balance_residual_N=(after["mass_kg"] * dv - impulse) / duration,
                 angular_balance_residual_Nm=(dl - angular_impulse) / duration,
                 unexpected_contacts_during_step=unexpected,
                 intent=np.asarray(action).copy(),
                 adapter_steering=np.asarray(env.control.steer).copy(),
                 rear_torque_Nm=np.asarray(command["rear_torque"]).copy(),
                 wheel_angular_velocity_radps=env._joint_velocities(env.spin_joints),
                 chassis_up_world=rotation[:, 2].copy())
    return after
