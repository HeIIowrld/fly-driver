import pytest

from scripts.audit_fly_mechanism_coverage import anatomical_inventory


def fixture():
    graph = dict(ids=['eye', 'dn', 'motor'], pre=[0, 1], post=[1, 2])
    constructor = dict(visual_ids=['eye'], output_groups=dict(steering=['dn']))
    annotations = [dict(bodyId='eye', type='T4a', superclass='optic'),
                   dict(bodyId='dn', type='DNa02', superclass='descending_neuron'),
                   dict(bodyId='motor', type='wingMN', superclass='vnc_motor', subclass='wm')]
    return graph, constructor, annotations


def test_cell_presence_does_not_validate_motor_or_visual_function():
    graph, constructor, annotations = fixture()
    result = anatomical_inventory(graph, constructor, annotations, ['T4a', 'T5a'])
    assert result['reference_visual_type_counts'] == dict(T4a=1, T5a=0)
    assert result['vnc_motor_neurons_present'] == 1
    assert result['wing_motor_annotation_candidates_present'] == 1
    assert result['wing_motor_annotation_candidates_read_out'] == 0
    assert result['output_annotations']['steering'][0]['superclass'] == 'descending_neuron'
    assert not result['physiological_visual_response_verified']
    assert not result['physiological_body_motor_response_verified']


def test_missing_or_duplicate_neuron_annotations_are_not_silently_filled():
    graph, constructor, annotations = fixture()
    for rows in (annotations[:-1], annotations+[annotations[0]]):
        with pytest.raises(ValueError):
            anatomical_inventory(graph, constructor, rows, ['T4a'])
