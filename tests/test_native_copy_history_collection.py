import json

import pytest

from scripts.record_ac_engine_ai_probe import finish_native_copy_history


class Sampler:
    def __init__(self, rows, error=None):
        self.rows, self.error, self.stopped = rows, error, False

    def stop(self):
        self.stopped = True


def test_history_is_saved_without_granting_training(tmp_path):
    rows = [{'authority': {'packet_id': 7}, 'copy': {'second_copy_end': 3.0}}]
    sampler = Sampler(rows)
    result = finish_native_copy_history(sampler, tmp_path)
    assert sampler.stopped and result['rows'] == 1
    assert result['fit_permission_granted'] is False and result['host_receipts_only'] is True
    assert json.loads((tmp_path/'native-copy-history.json').read_text()) == rows


def test_failed_sampler_preserves_partial_history_and_fails(tmp_path):
    sampler = Sampler([{'packet_id': 7}], 'mapping lost')
    with pytest.raises(RuntimeError, match='mapping lost'):
        finish_native_copy_history(sampler, tmp_path)
    assert sampler.stopped
    assert json.loads((tmp_path/'native-copy-history.json').read_text()) == sampler.rows
