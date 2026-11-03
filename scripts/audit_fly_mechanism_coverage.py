"""Inventory current anatomical coverage; cell presence never proves a functional circuit."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def anatomical_inventory(graph, constructor, annotations, visual_types):
    ids = [str(value) for value in graph['ids']]
    lookup = {str(row['bodyId']): row for row in annotations}
    if len(lookup) != len(annotations) or not set(ids) <= lookup.keys():
        raise ValueError('Unique original annotations for every graph neuron required')
    visual = [str(value) for value in constructor['visual_ids']]
    outputs = {key: [str(value) for value in values]
               for key, values in constructor['output_groups'].items()}
    if not set(visual) <= set(ids) or any(not set(v) <= set(ids) for v in outputs.values()):
        raise ValueError('Declared input/output IDs must belong to the actual graph')
    counts = Counter(lookup[key].get('type') for key in ids)
    return dict(
        nodes=len(ids), directed_connections=len(graph['pre']),
        visual_input_types=dict(Counter(lookup[key].get('type') for key in visual)),
        reference_visual_type_counts={name: counts[name] for name in visual_types},
        vnc_motor_neurons_present=sum(lookup[key].get('superclass') == 'vnc_motor' for key in ids),
        flight_descending_candidate_counts={
            name: sum(str(lookup[key].get('type', '')).startswith(name) for key in ids)
            for name in ('DNg02', 'DNp54')},
        wing_motor_annotation_candidates_present=sum(
            lookup[key].get('superclass') == 'vnc_motor'
            and lookup[key].get('subclass') == 'wm' for key in ids),
        wing_motor_annotation_candidates_read_out=sum(
            lookup[key].get('superclass') == 'vnc_motor'
            and lookup[key].get('subclass') == 'wm'
            for values in outputs.values() for key in values),
        output_annotations={name: [dict(body_id=key, type=lookup[key].get('type'),
                                      superclass=lookup[key].get('superclass'))
                                   for key in values] for name, values in outputs.items()},
        cell_presence_is_not_pathway_or_function_validation=True,
        physiological_visual_response_verified=False,
        physiological_body_motor_response_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--visual-config', required=True)
    parser.add_argument('--flyvis-source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.training_run)
    result = json.loads((root/'result.json').read_text(encoding='utf8'))
    if (result['status'] != 'completed'
            or sha256(root/'checkpoint.pt') != result['checkpoint_sha256']):
        raise ValueError('Completed hash-verified checkpoint required')
    checkpoint = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=True)
    asset = Path('assets/external/malecns-vehicle-candidate-20261027')
    registry = json.loads((asset/'source.json').read_text(encoding='utf8'))
    for name in ('vehicle-candidate.npz', 'annotations.json'):
        if sha256(asset/name) != registry['files_sha256'][name]:
            raise ValueError('Registered anatomical source changed')
    with load_array(asset/'vehicle-candidate.npz', allow_pickle=False) as original:
        if not all(key in original and np.array_equal(value, original[key])
                   for key, value in checkpoint['graph'].items()):
            raise ValueError('Checkpoint graph differs from measured-structure candidate')
    upstream = Path(args.flyvis_source)
    source = json.loads((upstream/'source.json').read_text(encoding='utf8'))
    name = 'flyvis/connectome/fib25-fib19_v2.2.json'
    if sha256(upstream/'upstream'/name) != source['files_sha256'][name]:
        raise ValueError('Pinned visual reference changed')
    visual = json.loads((upstream/'upstream'/name).read_text(encoding='utf8'))
    annotations = json.loads((asset/'annotations.json').read_text(encoding='utf8'))
    inventory = anatomical_inventory(checkpoint['graph'], checkpoint['constructor'],
                                    annotations, [node['name'] for node in visual['nodes']])
    run = begin_run(args.output, dict(arguments=vars(args), seed=863008, scope=__doc__,
        checkpoint_sha256=sha256(root/'checkpoint.pt'), annotation_sha256=sha256(
            asset/'annotations.json'), flyvis_source=source, actuator_messages_sent=0))
    configuration = json.loads(Path(args.visual_config).read_text(encoding='utf8'))
    write_json(run/'result.json', dict(status='completed', inventory=inventory,
        native_retina_declared_scope=configuration['native_retina']['scope'],
        visual_reference_integrated=False,
        motor_control_stage='current direct output-to-pedal codec',
        anatomical_ID_registration_between_datasets_verified=False,
        whole_fly_sensorimotor_mechanism_verified=False, closed_loop_evaluated=False,
        actuator_messages_sent=0, goal_achieved=False))


if __name__ == '__main__':
    main()
