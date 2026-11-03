import time

import pytest

from flybrain_drive.lifecycle_observation import LifecycleObserver


def test_success_and_failure_observations_are_not_silently_repaired():
    status = {'generation':'epoch-a', 'paused':True, 'ai_controlled':False}
    observer = LifecycleObserver('unused', 'epoch-a', reader=lambda path: (status, time.time()))
    observer.sample()
    assert observer.events[0]['fresh'] and observer.events[0]['same_generation']
    assert observer.events[0]['status']['paused'] is True
    assert len(observer.events[0]['canonical_payload_sha256']) == 64
    def failed(path): raise PermissionError('busy')
    observer.reader = failed
    observer.sample()
    assert observer.events[1]['error']['type'] == 'PermissionError'
    assert 'status' not in observer.events[1]


def test_interval_and_generation_contract():
    with pytest.raises(ValueError):
        LifecycleObserver('unused', '', interval=.01)
    with pytest.raises(ValueError):
        LifecycleObserver('unused', 'epoch-a', interval=.2)


def test_pause_is_signalled_only_after_fresh_matching_active_session():
    current = {'generation':'epoch-a', 'paused':True, 'ai_controlled':False, 'expert_probe_active':False}
    observer = LifecycleObserver('unused', 'epoch-a', reader=lambda path: (current.copy(), time.time()))
    observer.sample()
    assert not observer.pause_after_active.is_set()
    current.update(paused=False, ai_controlled=True, expert_probe_active=True)
    observer.sample()
    current.update(paused=True, ai_controlled=False, expert_probe_active=False, generation='other')
    observer.sample()
    assert not observer.pause_after_active.is_set()
    current['generation'] = 'epoch-a'
    observer.reader = lambda path: (current.copy(), time.time()-5)
    observer.sample()
    assert not observer.pause_after_active.is_set()
    observer.reader = lambda path: (current.copy(), time.time())
    observer.sample()
    assert observer.pause_after_active.is_set()
    assert observer.pause_observation is observer.events[-1]
