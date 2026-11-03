"""Receipt ordering must never make an unknown or future packet valid."""

from scripts.audit_group_availability_body import select_prior_packet


def test_first_packet_age_unknown_and_strict_boundary():
    receipts = [1.0, 2.0, 3.0]
    assert select_prior_packet(receipts, .9) is None
    assert select_prior_packet(receipts, 1.5) is None
    assert select_prior_packet(receipts, 2.0) is None
    assert select_prior_packet(receipts, 2.0001) == 1
    assert select_prior_packet(receipts, 3.0) == 1
    assert select_prior_packet(receipts, 3.1) == 2
