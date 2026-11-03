import numpy as np

from scripts.audit_body_visual_convergence_paths import distances_to_target, summarize


def test_reverse_bfs_respects_edge_direction_and_unreachable_nodes():
    # 0 -> 1 -> 3; 2 -> 3; 3 -> 4. Node 4 cannot reach 3.
    pre = np.array([0, 1, 2, 3])
    post = np.array([1, 3, 3, 4])
    distances = distances_to_target(5, pre, post, 3)
    np.testing.assert_array_equal(distances, [2, 1, 1, 0, -1])
    result = summarize(distances, np.array([0, 2, 4]))
    assert result['source_count'] == 3
    assert result['reachable_count'] == 2
    assert result['shortest_hops'] == 1
    assert result['hop_histogram']['2'] == 1
