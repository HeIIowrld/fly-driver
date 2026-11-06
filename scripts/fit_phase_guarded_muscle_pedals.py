"""TRAIN-only follow-up: protect speed bands and pedal phases as well as totals."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.joint_balance_muscle_readout import JointBalanceMusclePedalReadout
from flybrain_drive.muscle_interaction_candidate import load_muscle_interaction_head
from flybrain_drive.sac_unified_student import read
from scripts.fit_current_flight_launch_pedals import datasets, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = Path(args.base_fit)
    old = read(source / "manifest.json")["config"]
    cp = source / "pedals.pt"
    if sha256(cp) != read(source / "result.json")["checkpoint_sha256"]:
        raise ValueError("Sealed prior fit required")
    prior = torch.load(cp, weights_only=True, map_location="cpu")["state"]
    parent, _ = load_muscle_interaction_head(old["fixed_decoder"])
    data = datasets(Path(old["evaluation"]))
    train = data[:2]
    if any(x["receipt"]["role"] != "train" for x in train):
        raise ValueError("Exact TRAIN roles required")
    head = JointBalanceMusclePedalReadout(parent, prior["centers"], prior["width"])
    centers, width = head.centers.numpy().astype(np.float64), float(head.width)
    muscle = np.concatenate([item["muscles"] for item in train])
    reference = parent.base.training_reference.detach().numpy()
    scale = parent.base.training_scale.detach().numpy()
    z = ((muscle - reference) / scale).astype(np.float64)
    phi = np.exp(-.5 * np.sum((z[:, None, :4] - centers)**2, axis=-1) / width**2)
    x = np.column_stack((z, np.ones(len(z)), phi))
    y = np.concatenate([item["data"]["labels"][:, 1:] for item in train]).astype(np.float64)
    anchor = np.concatenate([item["original"][:, 1:] for item in train]).astype(np.float64)
    low = np.concatenate([item["low"] for item in train])
    origin = np.concatenate((parent.base.projection.weight.detach().numpy()[1:],
                             parent.base.projection.bias.detach().numpy()[1:, None],
                             np.zeros((2, len(centers)))), axis=1)
    # Start at the original head, not at a dev-selected iterate or previous learned output.
    groups, limits, totals = [], [], []
    start = 0
    for i, item in enumerate(train):
        speed = item["data"]["body"][:, 16]
        high = item["high"]
        for label, submask in (
                ("all_high", high), ("below_25", speed < 25),
                ("25_to_40", (speed >= 25) & (speed < 40)), ("40_plus", speed >= 40),
                ("braking", item["data"]["labels"][:, 2] > .1),
                ("high_throttle", item["data"]["labels"][:, 1] > .8)):
            indices = start + np.flatnonzero(high & submask)
            if len(indices) < 32:
                raise ValueError("Insufficient original samples in " + label)
            ratio = .98 if label == "all_high" else .995
            mse = np.mean((anchor[indices] - y[indices])**2, axis=0)
            groups.append((i, label, indices, ratio, mse))
            limits.extend(mse * ratio)
            if label == "all_high":
                totals.append(indices)
        start += len(speed)
    limits = np.array(limits)
    bounds = [(-10., 10.)] * origin.size
    for index in (4, 5):
        bounds[index] = (.0001, 10.)
        bounds[origin.shape[1] + index] = (-10., -.0001)
    objective_weights = low / low.sum()
    for indices in totals:
        objective_weights[indices] += .25 / (len(totals) * len(indices))
    objective_rows = np.flatnonzero(objective_weights)
    cost_weights = objective_weights[objective_rows, None]

    def predictions(vector):
        return expit(x @ vector.reshape(origin.shape).T)

    def objective(vector):
        p = predictions(vector)[objective_rows]
        e = p - y[objective_rows]
        diff = vector - origin.ravel()
        loss = np.sum(cost_weights * e**2) / 2 + .001 * np.mean(diff**2)
        grad = (e * p * (1-p) * cost_weights).T @ x[objective_rows]
        return loss, grad.ravel() + .002 * diff / len(diff)

    def constraints(vector):
        p = predictions(vector)
        e = p-y
        return limits - np.concatenate([np.mean(e[idx]**2, axis=0) for _, _, idx, _, _ in groups])

    def jacobian(vector):
        p = predictions(vector)
        slope = 2*(p-y)*p*(1-p)
        jac = np.zeros((len(limits), *origin.shape))
        for i, (_, _, idx, _, _) in enumerate(groups):
            grad = -(slope[idx].T @ x[idx]) / len(idx)
            jac[2*i, 0], jac[2*i+1, 1] = grad
        return jac.reshape(len(limits), -1)

    config = dict(old, base_fit=str(source), base_fit_checkpoint_sha256=sha256(cp),
                  seed=784106, optimizer="SLSQP with analytic derivatives; max500; independent TRAIN phase constraints",
                  source_sha256={**old["source_sha256"], __file__: sha256(__file__)},
                  phase_constraints=[dict(episode=i, name=name, labels=len(idx), ratio=ratio)
                                     for i, name, idx, ratio, _ in groups])
    out = begin_run(args.output, config)
    journal = []

    def record(vector):
        journal.append(dict(iteration=len(journal)+1, loss=float(objective(vector)[0]),
                            minimum_slack=float(constraints(vector).min()), parameters=vector.tolist()))

    opt = minimize(objective, origin.ravel(), method="SLSQP", jac=True, bounds=bounds,
                   constraints=[dict(type="ineq", fun=constraints, jac=jacobian)], callback=record,
                   options=dict(maxiter=500, ftol=1e-10))
    theta = torch.tensor(opt.x.reshape(origin.shape), dtype=torch.float32)
    with torch.no_grad():
        head.pedal_projection.weight[1:].copy_(theta[:, :6])
        head.pedal_projection.bias[1:].copy_(theta[:, 6])
        head.offset.weight.copy_(theta[:, 7:])
    head.verify()
    rows, outputs = metrics(head, data)
    pred = np.concatenate([outputs["prediction_0"][:, 1:], outputs["prediction_1"][:, 1:]])
    ratios = [(np.mean((pred[idx] - y[idx])**2, axis=0) / mse).tolist()
              for _, _, idx, _, mse in groups]
    feasible = bool(opt.success and constraints(opt.x).min() >= -1e-7 and
                    all(max(actual) <= group[3]+1e-5 for actual, group in zip(ratios, groups)))
    state = dict(centers=head.centers, width=head.width, offset=head.offset.state_dict(),
                 pedal_projection=head.pedal_projection.state_dict())
    torch.save(dict(schema="flybrain.balanced_muscle_pedals.experimental.v1", variant=old["variant"],
                    parent_checkpoint_sha256=old["parent_checkpoint_sha256"], actor_kind=old["actor_kind"],
                    state=state, actor_qualified=False), out / "pedals.pt")
    np.savez_compressed(out / "predictions.npz", **outputs)
    write_json(out / "training-journal.json", journal)
    write_json(out / "result.json", dict(
        status="completed_balanced_muscle_pedal_fit" if feasible else "failed_constraints",
        optimizer_success=bool(opt.success), optimizer_message=str(opt.message), iterations=int(opt.nit),
        checkpoint_sha256=sha256(out / "pedals.pt"), predictions_sha256=sha256(out / "predictions.npz"),
        metrics=rows, training_high_mse_ratios=[ratios[i] for i, g in enumerate(groups) if g[1]=="all_high"],
        training_phase_mse_ratios=ratios, minimum_train_slack=float(constraints(opt.x).min()),
        low_train_labels=int(low.sum()), actor_qualified=False, vehicle_control_sent=False,
        development_used_for_fit=False, source_circuit_and_steering_unchanged=True))


if __name__ == "__main__":
    main()
