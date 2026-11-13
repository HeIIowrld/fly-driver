"""Versioned virtual-eye mount; original dashboard recordings keep their original identity."""

import numpy as np

from .ac_csp_geometry import camera_geometry

EXTERIOR_FRONT_CONTRACT = dict(schema='flybrain.ac.camera_mount.v1',
    layout_id='exterior_front_v3', layout_revision=3, front_mount='bmw_z4_gt3_bumper_v1',
    front_offset_forward_m=2.5, front_offset_up_m=1., scene='full_scene', car_id='bmw_z4_gt3')

EXTERIOR_SIDE_120_CONTRACT = dict(EXTERIOR_FRONT_CONTRACT,
    layout_id='exterior_side120_v4', layout_revision=4, side_vertical_fov_deg=120)
EXTERIOR_CONTRACTS = {
    contract['layout_id']: contract
    for contract in (EXTERIOR_FRONT_CONTRACT, EXTERIOR_SIDE_120_CONTRACT)
}


def configured_mount(config):
    contract = config.get('camera_mount_contract')
    if contract is None:
        return None
    if contract not in EXTERIOR_CONTRACTS.values():
        raise ValueError('Exact explicit exterior-front camera mount contract required')
    return contract['layout_id']


def validate_frame_mount(frame, config):
    mount = configured_mount(config)
    if mount is None:
        if (frame.get('camera_layout_revision') != 2
                or frame.get('camera_mount_contract') is not None):
            raise ValueError('Legacy dashboard configuration cannot consume a different mount')
    else:
        contract = EXTERIOR_CONTRACTS[mount]
        if (frame.get('camera_layout_revision') != contract['layout_revision']
                or frame.get('camera_mount_contract') != contract
                or frame.get('car_id') != contract['car_id']):
            raise ValueError('Recorded camera mount differs from explicit exterior-front contract')
    geometry = camera_geometry(frame['views'])
    if mount is not None:
        expected_fov = EXTERIOR_CONTRACTS[mount].get('side_vertical_fov_deg', 90)
        if any(not np.isclose(geometry[side]['vertical_fov_degrees'], expected_fov,
                              atol=.001, rtol=0) for side in ('left', 'right')):
            raise ValueError('Returned exterior side FOV differs from declared mount')
        views = {v['view_id']: v for v in frame['views']}
        position = {k: np.asarray(v['renderer_only_world_position']) for k, v in views.items()}
        midpoint = (position['left']+position['right'])/2
        forward = np.asarray(views['front']['renderer_only_world_look'])
        left = np.asarray(views['left']['renderer_only_world_look'])
        # Same2mm returned native-float pose tolerance as camera_geometry; no fitted offsets.
        for actual, expected in [
                (position['front']-midpoint, 2.5*forward),
                (midpoint-position['rear'], 2.4*forward),
                (position['left']-position['right'], 2.3*left)]:
            if not np.allclose(actual, expected, atol=.002, rtol=0):
                raise ValueError('Returned exterior camera centers differ from declared mount')
