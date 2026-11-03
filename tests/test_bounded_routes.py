import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.sparse import csr_matrix

from flybrain_drive.bounded_routes import shortest_route_union


def graph():
    # Two equally short paths, a longer detour, and an unreachable source 6.
    pre = [0, 1, 0, 2, 0, 4, 5, 2]
    post = [1, 3, 2, 3, 4, 5, 3, 0]
    return csr_matrix((np.ones(len(pre)), (post, pre)), shape=(7, 7))


def test_union_keeps_all_equal_paths_but_not_longer_detours_or_unreachable_sources():
    nodes, pairs = shortest_route_union(graph(), [0, 6], [3], max_hops=4, max_nodes=7)
    assert nodes == [0, 1, 2, 3]
    assert pairs == [{"source_index": 0, "target_index": 3, "hops": 2,
                      "route_union_nodes": 4}]
    assert shortest_route_union(graph(), [3], [0], max_hops=4, max_nodes=7) == ([], [])
    assert shortest_route_union(graph(), [0], [3], max_hops=1, max_nodes=7) == ([], [])


def test_budget_failure_never_silently_selects_one_path():
    with pytest.raises(ValueError, match="no truncation"):
        shortest_route_union(graph(), [0], [3], max_hops=3, max_nodes=3)


@pytest.mark.parametrize("sources,targets,hops", [([0, 0], [3], 3), ([7], [3], 3),
                                               ([0], [], 3), ([0], [3], 0)])
def test_invalid_endpoints_and_bounds_are_rejected(sources, targets, hops):
    with pytest.raises(ValueError):
        shortest_route_union(graph(), sources, targets, max_hops=hops, max_nodes=7)
