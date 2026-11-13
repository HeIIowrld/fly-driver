"""Body-frame local racing line and speed envelope from predicted road sections."""

import numpy as np
from numpy.polynomial import Polynomial


def maneuver_from_line(line, speed_mps, *, lookahead_m, speed_bound=None):
    """Encode a local curve as desired left yaw rate and desired speed in SI."""
    if not np.isfinite([speed_mps, lookahead_m]).all() or speed_mps < 0 or lookahead_m <= 0:
        raise ValueError("Finite nonnegative speed and positive lookahead required")
    arc = line["arc_m"]
    x = np.interp(lookahead_m, arc, line["x_m"])
    y = np.interp(lookahead_m, arc, line["y_m"])
    target = line["speed_bound_mps"] if speed_bound is None else speed_bound
    # Pure pursuit is a geometric command adapter. Yaw feedback is applied
    # only within the physical flight circuit, where its availability is known.
    yaw = 2*y*speed_mps/(x*x+y*y)
    return np.array([np.clip(yaw, -2., 2.), target[0]], dtype=np.float32)


def speed_bound_with_corner_preview(line, distance_m, *, corner_speed_mps, braking_mps2):
    """Apply an engineering corner speed cap to an already selected local line.

    The forecast supplies a distance, not a turn radius or measured grip.
    Apply the cap from the predicted onset through the available local horizon.
    Pass None for a missing or abstaining forecast.
    """
    original = np.asarray(line["speed_bound_mps"], dtype=float)
    if distance_m is None:
        return original.copy()
    if (not np.isfinite([distance_m, corner_speed_mps, braking_mps2]).all()
            or distance_m < 0 or min(corner_speed_mps, braking_mps2) <= 0):
        raise ValueError("Finite nonnegative onset and positive speed and braking limits required")
    arc = np.asarray(line["arc_m"], dtype=float)
    approach = np.maximum(distance_m - arc, 0.)
    return np.minimum(original, np.sqrt(corner_speed_mps**2 + 2*braking_mps2*approach))


def plan_local_line(sections, speed_mps, *, clearance_m, max_curvature,
                    lateral_accel_mps2, braking_mps2, speed_ceiling_mps, refine_on_failure=False,
                    clearance_time_budget=None, corner_preview=None, prior_path=None):
    """Select a smooth tangent-matched path and return a braking-feasible speed bound.

    Clearance is relative to the interpolated predicted corridor. The first
    section bounds only the path at and beyond that section. Vehicle limits
    are engineering inputs and the result carries no actuator commands.
    Refine the endpoint grid from 9 to 17 points per axis only on request
    and only if the coarse grid finds no feasible curve. A positive fractional
    clearance_time_budget prioritizes minimum corridor clearance among curves
    whose time score stays within that fraction of the fastest candidate.
    A corner preview is (signed left direction, onset distance in metres).
    It can select a bounded deformation of the baseline candidate.
    A prior path in current body coordinates ranks feasible candidates within
    3% of the fastest envelope when at least 5 m of constrained path overlaps.
    """
    limits = np.asarray([clearance_m, max_curvature, lateral_accel_mps2,
                         braking_mps2, speed_ceiling_mps, speed_mps], dtype=float)
    if not np.isfinite(limits).all() or np.any(limits[:5] <= 0) or speed_mps < 0:
        raise ValueError("Finite positive limits and nonnegative speed required")
    if clearance_time_budget is not None and (
            not np.isfinite(clearance_time_budget) or clearance_time_budget < 0):
        raise ValueError("Clearance time budget must be a finite nonnegative fraction")
    if corner_preview is not None:
        direction, onset_m = corner_preview
        if direction not in (-1, 1) or not np.isfinite(onset_m) or onset_m < 0.:
            raise ValueError("Corner preview requires a signed direction and finite nonnegative onset")
    usable = []
    for section in sections:
        if not section["valid"]:
            break
        usable.append([section["forward_x_m"], section["right_y_m"], section["left_y_m"]])
    if len(usable) < 2:
        return None, "insufficient_contiguous_sections"
    x, right, left = np.asarray(usable, dtype=float).T
    if (not np.isfinite([x, right, left]).all() or x[0] <= 0
            or np.any(np.diff(x) <= 0) or np.any(left <= right)):
        raise ValueError("Finite ordered forward sections with left greater than right required")
    lower, upper = right + clearance_m, left - clearance_m
    if np.any(lower > upper):
        return None, "corridor_too_narrow"

    # A cubic starts at the ego origin with zero relative heading. Sample
    # lateral placements at the near and far sections and solve coefficients.
    near, far = x[[0, -1]]
    for grid_points in ((9, 17) if refine_on_failure else (9,)):
        fractions = np.linspace(0., 1., grid_points)
        endpoints = np.stack(np.meshgrid(lower[0]+fractions*(upper[0]-lower[0]),
                                         lower[-1]+fractions*(upper[-1]-lower[-1])), axis=-1).reshape(-1, 2)
        u0 = near / far
        a, b = np.linalg.solve([[u0**2, u0**3], [1., 1.]], endpoints.T)
        sample = np.unique(np.r_[np.linspace(0., far, int(np.ceil(far/.25))+1), x])
        u = sample / far
        y = a[:, None]*u**2 + b[:, None]*u**3
        slope = (2*a[:, None]*u+3*b[:, None]*u**2)/far
        curvature = (2*a[:, None]+6*b[:, None]*u)/far**2/(1+slope**2)**1.5
        # Check exact clearance extrema on each linear corridor segment.
        minimum_slack = np.full(len(a), np.inf)
        feasible = np.max(np.abs(curvature), axis=1) <= max_curvature
        for i in range(len(x)-1):
            lo, hi = x[i:i+2]/far
            for edge, sign in ((lower, 1.), (upper, -1.)):
                gradient = (edge[i+1]-edge[i])/(hi-lo)
                quadratic, linear = 3*b, 2*a
                discriminant = linear**2 + 4*quadratic*gradient
                # Stable quadratic roots, with separate handling for a linear derivative.
                q = -.5*(linear + np.copysign(np.sqrt(np.maximum(discriminant, 0.)), linear))
                root1 = np.full_like(a, np.nan)
                root2 = np.full_like(a, np.nan)
                real_quadratic = (quadratic != 0.) & (discriminant >= 0.)
                np.divide(q, quadratic, out=root1, where=real_quadratic)
                np.divide(-gradient, q, out=root2, where=real_quadratic & (q != 0.))
                np.divide(gradient, linear, out=root1, where=(quadratic == 0.) & (linear != 0.))
                # Invalid roots add a duplicate endpoint, which cannot change minimum clearance.
                check = np.column_stack((np.full_like(a, lo), np.full_like(a, hi),
                    np.where((lo < root1) & (root1 < hi), root1, lo),
                    np.where((lo < root2) & (root2 < hi), root2, lo)))
                distance = a[:, None]*check**2+b[:, None]*check**3-edge[i]-gradient*(check-lo)
                slack = np.min(sign*distance, axis=1)
                minimum_slack = np.minimum(minimum_slack, slack)
                feasible &= slack >= -1e-9
        if feasible.any():
            break
    else:
        return None, "no_feasible_curve"

    ids = np.flatnonzero(feasible)
    arc = np.column_stack((np.zeros(len(ids)), np.cumsum(np.hypot(np.diff(sample), np.diff(y[ids], axis=1)), axis=1)))
    cap_squared = np.minimum(speed_ceiling_mps**2, lateral_accel_mps2/np.maximum(np.abs(curvature[ids]), 1e-12))
    # Backward propagation of braking bounds along each candidate's arc.
    cap_squared = np.minimum.accumulate((cap_squared+2*braking_mps2*arc)[:, ::-1], axis=1)[:, ::-1]-2*braking_mps2*arc
    speed = np.sqrt(cap_squared)
    time_score = np.sum(2*np.diff(arc, axis=1)/(speed[:, 1:]+speed[:, :-1]), axis=1)
    best = int(np.argmin(time_score))
    if clearance_time_budget:
        eligible = np.flatnonzero(time_score <= time_score[best]*(1.+clearance_time_budget))
        order = np.lexsort((time_score[eligible], -minimum_slack[ids[eligible]]))
        best = int(eligible[order[0]])
    if prior_path is not None:
        prior_x, prior_y = np.asarray(prior_path, dtype=float).T
        if (len(prior_x) < 2 or not np.isfinite([prior_x, prior_y]).all()
                or np.any(np.diff(prior_x) <= 0)):
            raise ValueError("Prior path requires finite ordered body-frame points")
        overlap = (sample >= max(near, prior_x[0])) & (sample <= prior_x[-1])
        if overlap.any() and np.ptp(sample[overlap]) >= 5.:
            eligible = np.flatnonzero(time_score <= time_score[best]*1.03)
            target = np.interp(sample[overlap], prior_x, prior_y)
            deviation = np.mean((y[ids[eligible]][:, overlap]-target)**2, axis=1)
            best = int(eligible[np.lexsort((time_score[eligible], deviation))[0]])
    j = ids[best]
    result = dict(x_m=sample.tolist(), y_m=y[j].tolist(), arc_m=arc[best].tolist(),
                  curvature_inv_m=curvature[j].tolist(), speed_bound_mps=speed[best].tolist(),
                  first_constrained_x_m=float(near), envelope_time_score_s=float(time_score[best]))
    if grid_points > 9:
        result["candidate_grid_points_per_axis"] = grid_points
    if corner_preview is not None:
        direction, onset_m = corner_preview
        centerline = .5*(right+left)
        center_arc = np.r_[0.,np.cumsum(np.hypot(np.diff(np.r_[0.,x]),np.diff(np.r_[0.,centerline])))]
        if center_arc[1]<=onset_m and onset_m+4.<=center_arc[-1]:
            stations = np.interp([onset_m,onset_m+4.],center_arc,np.r_[0.,x])
            local = stations/far
            split = float(local.mean())
            basis = Polynomial([0.,0.,-split,1.+split,-1.])
            scale = max(abs(basis(local)))
            curve = Polynomial([0.,0.,a[j],b[j]])
            midpoint = np.interp(stations,x,centerline)
            half_width = .5*np.interp(stations,x,upper-lower)
            target = midpoint + direction*np.array([-.30,.30])*half_width
            choices = []
            for amplitude in (.025,.05,.10,.20,.35,.50):
                candidate = curve + (direction*amplitude/scale)*basis
                movement = (candidate(local)-curve(local))*direction*np.array([-1.,1.])
                if np.any(movement<.01):
                    continue
                bounded = True
                for k in range(len(x)-1):
                    lo,hi = x[k:k+2]/far
                    for edge, sign in ((lower,1.),(upper,-1.)):
                        gradient = (edge[k+1]-edge[k])/(hi-lo)
                        gap = sign*(candidate-Polynomial([edge[k]-gradient*lo,gradient]))
                        roots = gap.deriv().roots()
                        critical = [z.real for z in roots if abs(z.imag)<1e-8 and lo<z.real<hi]
                        bounded &= bool(np.min(gap(np.array([lo,hi,*critical])))>=-1e-9)
                if not bounded:
                    continue
                cy = candidate(u)
                cs = candidate.deriv()(u)/far
                ck = candidate.deriv(2)(u)/far**2/(1.+cs**2)**1.5
                d1, d2, d3 = (candidate.deriv(k)/far**k for k in (1, 2, 3))
                extrema = (d3*(1.+d1*d1)-3.*d1*d2*d2).roots()
                critical = np.array([0., 1., *(z.real for z in extrema
                                               if abs(z.imag)<1e-9 and 0.<z.real<1.)])
                if np.max(abs(d2(critical)/(1.+d1(critical)**2)**1.5))>max_curvature:
                    continue
                ca = np.r_[0.,np.cumsum(np.hypot(np.diff(sample),np.diff(cy)))]
                cap = np.minimum(speed_ceiling_mps**2,lateral_accel_mps2/np.maximum(abs(ck),1e-12))
                cap = np.minimum.accumulate((cap+2*braking_mps2*ca)[::-1])[::-1]-2*braking_mps2*ca
                cv = np.sqrt(cap)
                score = float(np.sum(2*np.diff(ca)/(cv[1:]+cv[:-1])))
                if score<=time_score[best]*1.03:
                    error = float(np.sum(((candidate(local)-target)/np.maximum(half_width,.25))**2))
                    choices.append((error,score,cy,ca,ck,cv))
            if choices:
                _,score,cy,ca,ck,cv = min(choices,key=lambda choice:choice[:2])
                result.update(y_m=cy.tolist(),arc_m=ca.tolist(),curvature_inv_m=ck.tolist(),
                              speed_bound_mps=cv.tolist(),envelope_time_score_s=score)
    return result, "above_entry_speed_bound" if speed_mps > result["speed_bound_mps"][0] else "within_entry_speed_bound"
