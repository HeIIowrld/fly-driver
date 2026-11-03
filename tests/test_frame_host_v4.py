"""Private CPU files, synthetic pixels/futures/clocks only; no live/GPU/device calls."""

import copy
import hashlib
import json
import runpy
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_drive import collection_cleanup_v4 as cleanup
from flybrain_drive.ac_camera_archive import bundle_camera_files
from flybrain_drive.ac_csp_vision_host_v4 import copy_shared_group
from flybrain_drive.ac_metadata_mmf import encode_packet
from flybrain_drive.frame_persistence_profile_v3 import BufferedFrameProfile
from flybrain_drive.frame_persistence_profile_v4 import (
    EVENTS_PER_GROUP,
    PHASES,
    TRACE_ROW,
    TRACE_ROWS,
    HostCopyProfile,
    HostFrameWriterFactory,
    HostImmutableFrameWriter,
    HostPayloadNamespace,
    HostTrace,
    verify_host_persistence_archive,
)
from flybrain_drive.immutable_frame_writer import ImmutableFrameWriter
from scripts.prepare_frame_host_v4 import BASES, build

REQUEST = 'mmf-stream-'+'a'*32
VIEWS = ('front','left','right','rear')


def namespace(tmp_path):
    client = tmp_path/REQUEST
    client.mkdir()
    return HostPayloadNamespace.create(client,REQUEST,'g',1200)


def trace():
    wall,cpu = iter(range(1000,1000000)),iter(range(2000,1001000))
    return HostTrace(wall=lambda:next(wall),cpu=lambda:next(cpu))


def views(index=0):
    return [dict(view_id=n,file=f'{index:04d}-{n}.bmp',image_writer='host_lossless_bmp',
                 shared_texture_handle=str(i+10))for i,n in enumerate(VIEWS)]


def test_explicit_profile_preserves32_caps_and_all_freshness_gates():
    record = HostCopyProfile().record()
    assert record['persistence'] == BufferedFrameProfile().record()
    assert record['original_camera_freshness_seconds'] == .15
    assert record['original_actor_interval_seconds'] == .08
    assert record['original_heartbeat_age_seconds'] == 2.
    assert record['trace_maximum_binary_bytes'] == TRACE_ROWS*TRACE_ROW.size <= 8*1024**2
    for key in ('original_camera_freshness_seconds','trace_maximum_groups'):
        changed = copy.deepcopy(record)
        changed[key] *= 2
        with pytest.raises(ValueError):
            HostCopyProfile.from_record(changed)


def test_existing_namespace_is_never_reused_or_renamed(tmp_path):
    value = namespace(tmp_path)
    (value.directory/'foreign.bmp').write_bytes(b'original')
    with pytest.raises(FileExistsError):
        HostPayloadNamespace.create(tmp_path/REQUEST,REQUEST,'g',1200)
    assert (value.directory/'foreign.bmp').read_bytes() == b'original'
    assert value.record()['foreign_same_user_injection_excluded'] is False


@pytest.mark.parametrize('change',['duplicate','out_of_order','path','wrong_view','case'])
def test_ram_targets_are_once_only_canonical_and_no_retry(tmp_path,change):
    value = namespace(tmp_path)
    current = views()
    index = 0
    if change == 'duplicate':
        value.reserve(0,current)
    elif change == 'out_of_order':
        index,current = 1,views(1)
    elif change == 'path':
        current[0]['file'] = '../0000-front.bmp'
    elif change == 'wrong_view':
        current[0]['view_id'] = 'rear'
    else:
        current[0]['file'] = '0000-FRONT.bmp'
    with pytest.raises(ValueError):
        value.reserve(index,current)


def test_new_hot_path_has_no_path_exists_and_no_extra_read_or_ACK(tmp_path,monkeypatch):
    owner,timing,calls = namespace(tmp_path),trace(),[]
    rgb = np.zeros((180,320,3),np.uint8)
    class GPU:
        def read_group(self,handles,*,host_trace,group_id):
            calls.append(('copy',tuple(handles)))
            for phase in ('handles_parse_enter','handles_parse_exit',
                          'group_copy_enter','group_copy_exit'):
                host_trace.mark(group_id,phase)
            return [(rgb.copy(),dict(started_at=1.,finished_at=2.))for _ in range(4)]
    def no_fs(*args,**kwargs):
        pytest.fail('No pre-copy filesystem existence query in explicit host profile')
    monkeypatch.setattr(Path,'exists',no_fs)
    frame = dict(index=0,request_id=REQUEST,generation='g',views=views())
    pixels,receipts,ack = copy_shared_group(GPU(),frame,owner.directory,batched=True,
        host_namespace=owner,host_trace=timing,host_clock=lambda:3.,
        ack_writer=lambda value:calls.append(('ack',value['capture_id'])))
    assert calls == [('copy',('10','11','12','13')),('ack',0)]
    assert set(pixels) == set(VIEWS) and all(not v.flags.writeable for v in pixels.values())
    assert ack['capture_id'] == 0 and len(receipts) == 4
    with pytest.raises(ValueError):
        copy_shared_group(GPU(),frame,owner.directory,batched=True,
            host_namespace=owner,host_trace=timing,ack_writer=lambda a:pytest.fail('No ACK'))
    assert len(calls) == 2


def test_trace_is_preallocated_bounded_single_producer_and_never_sensor_clock():
    value = trace()
    assert len(value.data) == TRACE_ROWS*TRACE_ROW.size
    for _ in range(EVENTS_PER_GROUP):
        value.mark(0,'capacity_enter')
    with pytest.raises(ValueError):
        value.mark(0,'capacity_enter')
    assert value.count == EVENTS_PER_GROUP
    record = value.record()
    assert record['source_receipts_changed'] is False
    assert record['records'][0][2:] == [1000,2000]
    value.producer = -1
    with pytest.raises(ValueError):
        value.mark(1,'capacity_enter')


def test_actual_writer_persist_implementation_xb_and_singleproducer_are_unchanged(tmp_path):
    owner,timing = namespace(tmp_path),trace()
    assert HostImmutableFrameWriter._persist is ImmutableFrameWriter._persist
    writer = HostFrameWriterFactory(HostCopyProfile()).bind(owner,timing)(
        owner.directory,tmp_path/'stream.jsonl',mmf_metadata=True)
    files = [(f'0000-{name}.bmp',name.encode())for name in VIEWS]
    metadata = [('frame-0000.json',b'{}'),('ack-0000.json',b'{}'),('mmf-0000.mmf',b'raw')]
    (owner.directory/files[0][0]).write_bytes(b'foreign-original')
    writer.check_capacity()
    owner.reserve(0,views())
    writer.submit(0,files,b'original-line\n',metadata_files=metadata)
    with pytest.raises(RuntimeError):
        writer.close()
    assert writer.closed and (owner.directory/files[0][0]).read_bytes() == b'foreign-original'
    assert writer.statistics()['failed'] == 1


def test_three_root_bundle_preserves_flat_names_and_rejects_collision(tmp_path):
    roots = [tmp_path/name for name in ('producer','client','payload')]
    for root in roots:
        root.mkdir()
    for root,name,raw in zip(roots,('result.json','namespace.json','0000-front.bmp'),
                            (b'{}',b'{}',b'synthetic-image'),strict=True):
        (root/name).write_bytes(raw)
    result = bundle_camera_files(roots,tmp_path/'archive')
    assert result['files'] == 3 and result['all_hashes_verified']
    assert {r['file']for r in json.loads((tmp_path/'archive/archive.json').read_bytes())} == {
        'result.json','namespace.json','0000-front.bmp'}
    (roots[2]/'result.json').write_bytes(b'foreign')
    with pytest.raises(ValueError,match='collision'):
        bundle_camera_files(roots,tmp_path/'collision-archive')


def host_archive(tmp_path,monkeypatch):
    fixture = runpy.run_path('tests/test_frame_persistence_profile_v3.py')
    journal,row = fixture['valid_archive'](tmp_path,monkeypatch)
    # Synthetic metadata fixture only: replace its test-only ID consistently.
    frame = json.loads((tmp_path/'frame-0000.json').read_bytes())
    ack = json.loads((tmp_path/'ack-0000.json').read_bytes())
    frame['request_id'] = ack['request_id'] = REQUEST
    raw_frame,raw_ack = json.dumps(frame).encode(),json.dumps(ack).encode()
    slots = encode_packet(0,raw_frame)+encode_packet(0,raw_ack)
    frame_sha = hashlib.sha256(raw_frame).hexdigest()
    row['file_sha256'] = row['metadata_mmf_receipt']['raw_sha256'] = frame_sha
    row['metadata_mmf_receipt']['slots_sha256'] = hashlib.sha256(slots).hexdigest()
    data = {'frame-0000.json':raw_frame,'ack-0000.json':raw_ack,'mmf-0000.mmf':slots}
    for name,raw in data.items():
        (tmp_path/name).write_bytes(raw)
    line = (json.dumps(row)+'\n').encode()
    (tmp_path/'stream.jsonl').write_bytes(line)
    record = journal['records'][0]
    for file in record['files']:
        if file['file'] in data:
            raw = data[file['file']]
            file.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    record['stream_line_sha256'] = hashlib.sha256(line).hexdigest()
    size = sum(f['bytes']for f in record['files'])+len(line)
    journal['payload_admissions'][0].update(payload_bytes=size,retained_payload_bytes_after_submit=size)
    journal['maximum_retained_payload_bytes_observed'] = size
    (tmp_path/'frame-persistence-journal.json').write_text(json.dumps(journal))
    owner = namespace(tmp_path)
    owner.reserve(0,views())
    timing = trace()
    for phase in ('capacity_enter','capacity_exit',*PHASES[6:],'capacity_enter','capacity_exit'):
        timing.mark(0,phase)
    (tmp_path/'host-payload-namespace-v4.json').write_text(json.dumps(owner.record()))
    (tmp_path/'host-precopy-trace-v4.json').write_text(json.dumps(timing.record()))
    return fixture['Archive'](tmp_path),row


def test_strict_archive_checks_allpayload_plus_namespace_trace(tmp_path,monkeypatch):
    archive,row = host_archive(tmp_path,monkeypatch)
    assert verify_host_persistence_archive(archive,expected_mmf=True)[0] == row
    data = archive.json('host-precopy-trace-v4.json')
    # Count/monotonic clocks alone are insufficient.
    data['records'][2][1] = PHASES.index('ack_finished')
    (tmp_path/'host-precopy-trace-v4.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='trace order'):
        verify_host_persistence_archive(archive,expected_mmf=True)


def test_namespace_cannot_forge_reserved_groups_or_generation(tmp_path,monkeypatch):
    archive,_ = host_archive(tmp_path,monkeypatch)
    data = archive.json('host-payload-namespace-v4.json')
    data['reserved_groups'] = [0,1]
    (tmp_path/'host-payload-namespace-v4.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_host_persistence_archive(archive,expected_mmf=True)


def healthy_context():
    return SimpleNamespace(state='running',error=None,stop_event=threading.Event(),
        lease=SimpleNamespace(registration={'deadline_qpc_ns':160_000_000_000}),writer=object(),
        poll_thread=SimpleNamespace(is_alive=lambda:True),write_thread=SimpleNamespace(is_alive=lambda:True))


def clean_receipt():
    return dict(schema=cleanup.SCHEMA,complete=True,control_process_created=True,
        control_process_pid=12,control_process_returncode=1,control_process_terminal=True,
        camera=dict.fromkeys(('camera_thread_joined','producer_terminal_known','persistence_joined',
                              'precompute_joined','shared_reader_closed'),True))


@pytest.mark.parametrize('mutation', ['stopped','context_error','worker_dead','budget',
    'cleanup_unknown','not_running','false_control_terminal','missing_returncode'])
def test_uncertain_stopped_or_failed_context_never_retires_or_restarts(mutation,monkeypatch):
    context,receipt = healthy_context(),clean_receipt()
    if mutation == 'stopped':
        context.stop_event.set()
    elif mutation == 'context_error':
        context.error = 'original-error'
    elif mutation == 'worker_dead':
        context.poll_thread.is_alive = lambda:False
    elif mutation == 'budget':
        context.lease.registration['deadline_qpc_ns'] = 107_999_999_999
    elif mutation == 'cleanup_unknown':
        receipt['camera']['persistence_joined'] = False
    elif mutation == 'not_running':
        context.state = 'failed_fenced'
    elif mutation == 'false_control_terminal':
        receipt['control_process_terminal'] = False
    else:
        receipt['control_process_returncode'] = None
    monkeypatch.setattr(cleanup,'finish_owned_transport',lambda *a:pytest.fail('No retirement'))
    result = dict(status='failed',reason='original data failure')
    assert not cleanup.retire_clean_failed_data(context,None,None,None,result,
        dict(status='failed',owned_cleanup=receipt),now_qpc_ns=100_000_000_000)
    assert result['status'] == 'failed' and result['fit_allowed'] is False


def test_clean_data_failure_can_retire_same_healthy_transport_but_remains_failed(monkeypatch):
    context,result,calls = healthy_context(),dict(status='failed',reason='original'),[]
    def finish(actual,*args):
        assert actual is context and not context.stop_event.is_set()
        calls.append('same-current-context')
        context.state = 'closed'
        return dict(same_writer_joined=True,lease_removed=True)
    monkeypatch.setattr(cleanup,'finish_owned_transport',finish)
    assert cleanup.retire_clean_failed_data(context,None,None,None,result,
        dict(status='failed',owned_cleanup=clean_receipt()),now_qpc_ns=100_000_000_000)
    assert calls == ['same-current-context'] and result['status'] == 'failed'
    assert result['reason'] == 'original' and result['data_outcome'] == 'failed'
    assert result['transport_outcome'] == 'normally_retired_after_clean_failed_data'


def test_fresh_idle_failure_retains_failure_fence_without_clearing_context_error(monkeypatch):
    context,result = healthy_context(),dict(status='failed',reason='original')
    def finish(*args):
        raise ValueError('cleanup idle unknown')
    monkeypatch.setattr(cleanup,'finish_owned_transport',finish)
    assert not cleanup.retire_clean_failed_data(context,None,None,None,result,
        dict(status='failed',owned_cleanup=clean_receipt()),now_qpc_ns=100_000_000_000)
    assert result['status'] == 'failed' and context.state == 'running'
    assert result['transport_outcome'] == 'retirement_failed_fence_retained'


def test_generated_sources_are_exact_and_all_frozen_defaults_are_unchanged():
    assert all(path.read_bytes() == raw for path,raw in build().items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in BASES.items())
    original = Path('scripts/audit_ac_engine_ai_probe_buffered.py').read_text(encoding='utf8')
    current = Path('scripts/audit_ac_engine_ai_probe_host_v4.py').read_text(encoding='utf8')
    start,end = '        for index, event in enumerate(events):','    intervals = np.diff('
    old_block = original[original.index(start):original.index(end)]
    assert old_block == current[current.index(start):current.index(end)]
    source = Path('scripts/qualify_session_heartbeat_evidence_host_v4.py')
    runner = source.read_text(encoding='utf8')
    assert (runner.index('retire_clean_failed_data(context')
            < runner.index('context.stop_event.set()'))
    assert 'raise  # Data failure stays exit1' in runner


def test_standard_retirement_order_uses_original_context_and_no_stop_before_final(monkeypatch):
    from flybrain_drive import session_heartbeat_terminal_v2 as terminal

    calls = []
    context = healthy_context()
    context.writer = SimpleNamespace(last_committed_sequence=1)
    context.last_ack = (2,dict(phase='final',final_sequence=1))
    def retire():
        assert not context.stop_event.is_set()
        calls.append('retire')
    def fence(*,join_timeout):
        assert join_timeout == 2. and calls[-1] == 'final'
        context.stop_event.set()
        calls.append('same-worker-fence')
    def complete():
        assert calls[-1] == 'capture-sync'
        context.state = 'closed'
        calls.append('complete-sync-release')
        return dict(same_writer_joined=True,lease_removed=True)
    context.request_retire,context.fence_writer,context.complete_sync_return = retire,fence,complete
    def idle():
        calls.append('idle')
        return True
    def wait(actual,predicate,seconds,description):
        assert actual is context and predicate()
        if 'Final quiescent' in description:
            calls.append('final')
    def sync(actual):
        assert actual is context and context.stop_event.is_set()
        calls.append('sync-ACK')
        return True
    monkeypatch.setattr(terminal,'observe_sync_return',sync)
    evidence = SimpleNamespace(capture_sync=lambda c:calls.append('capture-sync'),
                               bind_handoff=lambda h:calls.append('bind-handoff'))
    handoff = cleanup.finish_owned_transport(context,evidence,idle,wait)
    assert handoff['lease_removed'] and calls == ['idle','retire','final','same-worker-fence',
        'sync-ACK','capture-sync','complete-sync-release','bind-handoff','idle']


def test_cleanup_proof_is_measured_not_inferred_from_missing_still_live_field():
    camera = dict.fromkeys(clean_receipt()['camera'],True)
    child = SimpleNamespace(pid=12,returncode=1,poll=lambda:1)
    vision = SimpleNamespace(owned_cleanup=camera,request_id=REQUEST)
    assert cleanup.recorder_cleanup(child,vision)['complete']
    camera['precompute_joined'] = False
    assert not cleanup.recorder_cleanup(child,vision)['complete']
    assert not cleanup.healthy_failed_data_eligible(healthy_context(),
        dict(clean_receipt(),camera=None),now_qpc_ns=100_000_000_000)
