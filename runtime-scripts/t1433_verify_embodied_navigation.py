"""Real fixed flight graph: initial identity and both learned command gradient paths."""
import base64
import json
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix

read = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
base = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
data = base/'20261029-t1360-v2-flight-episode-cache'
source = read(base/'20261029-t1361-v2-visual-boundary-training/result.json')
assert sha256(source['checkpoint']) == source['checkpoint_sha256']
run = begin_run(base/'20261029-t1433-embodied-navigation-path-verification',
    dict(scope=__doc__, program_sha256=sha256(__file__),
         source_checkpoint_sha256=source['checkpoint_sha256'],
         body_contract_sha256=sha256('configs/ac-virtual-navigation-body-v2.json'),
         actual_original_row_used_as_component_input=True,
         original_chronological_episode_replay_claimed=False))
(run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
torch.set_num_threads(1)
torch.manual_seed(863432)
assert torch.cuda.is_available()
entry = next(e for e in read(data/'dataset.json')['episodes'] if e['role'] == 'train')
with load_array(data/entry['file'], allow_pickle=False) as archive:
    arrays = {k: archive[k].copy() for k in archive.files}
selected = np.flatnonzero(arrays['target_valid'].all(1) & (
    arrays['observation_time']-arrays['observation_time'][0] >= 3))[0]
root = Path(entry['original']['recording'])
events = [json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf8').splitlines()]
event = events[selected]
assert arrays['observation_time'][selected] == event['decision_monotonic_seconds']
encoder = NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',
    native_identity=read(root/'identity.json'))
sample = decode_navigation_body_prefix(base64.b64decode(event['native']['prefix_base64'], validate=True),
    event['native']['received_monotonic_seconds'])
observation = encoder.encode(sample, episode_id=root.name,
    decision_monotonic_seconds=event['decision_monotonic_seconds'])
np.testing.assert_array_equal(observation.values[:16].astype(np.float32), arrays['body'][selected])
np.testing.assert_array_equal(observation.valid[:16], arrays['body_valid'][selected])
payload = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
model = EmbodiedVisualFlightCircuit('configs/flight-circuit-visual-v6-20261029.json', payload).to('cuda')
inputs = {k: torch.tensor(arrays[k][selected:selected+1], device='cuda') for k in
          ('visual', 'visual_valid', 'goals', 'scene_features')}
body = torch.tensor(observation.values[None].astype(np.float32), device='cuda')
valid = torch.tensor(observation.valid[None].copy(), device='cuda')
drive = model._drive_tensor(inputs['visual'], inputs['visual_valid'], body, valid,
                             inputs['goals'], inputs['scene_features'])
with torch.no_grad():
    legacy = model.backbone._drive_tensor(inputs['visual'], inputs['visual_valid'], body[:, :16],
        valid[:, :16], inputs['goals'], inputs['scene_features'])
assert torch.equal(drive, legacy), 'Zero boundary must preserve exact inherited input drive'
state = model.initial_state(1)
for _ in range(100):
    state = model._advance_elapsed_tensor(drive, state, .01)
intent = model._read_state_tensor(state)[0]
target = torch.tensor(arrays['target'][selected:selected+1], device='cuda')
loss = (intent-target).square().mean()
loss.backward()
turn = model.boundary.turn_output.weight.grad
power = model.boundary.power_output.weight.grad
assert turn is not None and power is not None and turn.abs().max() > 0 and power.abs().max() > 0
assert torch.isfinite(turn).all() and torch.isfinite(power).all()
assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
model.backbone.assert_body_boundary_protected()
assert sha256(source['checkpoint']) == source['checkpoint_sha256']
write_json(run/'spec.json', model.spec)
torch.save(dict(schema=model.schema, model=model.state_dict(), spec=model.spec,
    goal_contract=payload['goal_contract'], source_checkpoint=source['checkpoint'],
    source_checkpoint_sha256=source['checkpoint_sha256'], glutamate_sign=payload['glutamate_sign'],
    learned_optimizer_updates=0, actor_qualified=False), run/'initial-model.pt')
result = dict(status='passed_exact_initial_identity_and_real_graph_joint_gradients',
    original_row=int(selected), source_recording=str(root), original_row_clock=float(arrays['observation_time'][selected]),
    old16_float32_cache_values_and_masks_exact=True, vehicle_feedback_values=observation.values[16:].tolist(),
    initial_drive_exact=True, turn_head_maximum_gradient=float(turn.abs().max()),
    power_head_maximum_gradient=float(power.abs().max()),
    original_graph_body_and_readout_parameters_frozen=True,
    initial_model_sha256=sha256(run/'initial-model.pt'),
    all_original_source_weights_unchanged=True, optimizer_updates=0,
    single_held_input_component_test_not_driving_or_chronological_training=True,
    ui_actions_sent=0, actuator_messages_sent=0, goal_achieved=False)
write_json(run/'result.json', result)
print(json.dumps(result), flush=True)
