"""Preparation-only equivalence proof; no discovered module is imported/executed."""

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

from flybrain_drive.source_closure_inventory_v1 import SourceInventory


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-proof', required=True)
    parser.add_argument('--legacy-manifest', required=True)
    parser.add_argument('--new-proof', required=True)
    parser.add_argument('--new-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    own_files = ['src/flybrain_drive/source_closure_inventory_v1.py',
                 'scripts/prepare_source_closure_inventory_v1.py',
                 'tests/test_source_closure_inventory_v1.py']
    before = {p: sha(Path(p).read_bytes()) for p in own_files}
    (out/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    inputs = {}

    def load(name):
        path = Path(getattr(args, name))
        raw = path.read_bytes()
        inputs[name] = dict(path=str(path), sha256=sha(raw), size_bytes=len(raw))
        return json.loads(raw)

    old = load('legacy_proof')
    old_manifest = load('legacy_manifest')
    new = load('new_proof')
    new_manifest = load('new_manifest')
    checks = []

    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))

    # Only literal AST data from the old module, not its slow module-level walk.
    candidate = Path('src/flybrain_drive/pedal_train_candidate_host_v4.py').read_bytes()
    literal_seeds = next(ast.literal_eval(node.value) for node in ast.parse(candidate).body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'SOURCE_SEEDS'
                                 for t in node.targets))
    check('actual_frozen_legacy_seed_literals_match_saved_proof',
          tuple(old['seeds']) == literal_seeds)
    check('legacy_candidate_hash_matches_sealed_manifest',
          sha(candidate) == old_manifest['before'][
              'src/flybrain_drive/pedal_train_candidate_host_v4.py'])
    started = time.perf_counter()
    inventory = SourceInventory.capture(Path.cwd())
    captured = time.perf_counter()
    old_closure = inventory.local_import_closure(old['seeds'])
    old_done = time.perf_counter()
    new_closure = inventory.local_import_closure(new['seeds'])
    new_done = time.perf_counter()
    verification = inventory.verify_unchanged()
    checked = time.perf_counter()
    check('legacy_AST_closure_exact_saved_set', old_closure == tuple(sorted(old['closure'])))
    old_complete = set(old_closure) | set(old['configuration_reference_bindings'])
    check('legacy_explicit_bindings_preserved_77',
          len(old_complete) == old['mandatory_inputs'] == 77)
    check('new_closure_exact_saved_83', set(new_closure) == set(new['mandatory_source_sha256'])
          and len(new_closure) == 83)
    check('new_manifest_closure_exact',
          set(new_closure) == set(new_manifest['mandatory_new_AST_closure']))
    check('old_complete_is_new_closure_subset', old_complete <= set(new_closure))
    current = inventory.source_sha256
    for name in sorted(p for p in old_complete if p.endswith('.py')):
        check('legacy_source_exact:'+name, current[name] == old_manifest['before'][name])
    for name in sorted(p for p in new_closure if p.endswith('.py')):
        check('new_source_exact:'+name, current[name] == new['mandatory_source_sha256'][name]
              == new_manifest['before'][name])
    after = {p: sha(Path(p).read_bytes()) for p in own_files}
    check('new_helper_script_tests_unchanged_during_proof', before == after)
    check('no_Torch_or_discovered_packages_imported',
          not any(n == 'torch' or n.startswith('torch.') for n in sys.modules))
    failed = [c for c in checks if not c['passed']]
    result = dict(
        schema='flybrain.source_closure_inventory.preparation.v1',
        status='passed_saved_legacy_and_new_closure_equivalence' if not failed else 'failed',
        checks=checks, failed_checks=len(failed), check_count=len(checks), inputs=inputs,
        legacy_ast_closure=list(old_closure), legacy_complete_bindings=sorted(old_complete),
        new_closure=list(new_closure), source_sha256=current, source_before=before,
        source_after=after, inventory=list(sorted(inventory.python_paths)),
        inventory_verification=verification,
        timing_seconds=dict(initial_inventory=captured-started, old_closure=old_done-captured,
                            new_closure_shared_inventory=new_done-old_done,
                            seal_reinventory_and_reread=checked-new_done,
                            total=checked-started),
        mechanism='One scoped rg inventory shared by two closures; pure membership for all '
                  'module aliases, one cached AST read per actual reachable path; one explicit '
                  'seal reinventory/reread. Optional exact root scripts.py read preserves '
                  'legacy script module-before-package semantics.',
        limitations=[
            'No universal symlink/junction namespace equivalence is claimed. rg does not '
            'follow links here; exact sealed legacy/new closure comparisons are the adoption gate.',
            'Rechecks detect observed path/read drift, not a filesystem transaction '
            'or reverted edits.',
            'Non-Python bindings are retained verbatim, not opened or independently rehashed.',
            'Observed old stat/is_file waiting does not uniquely establish a NAS/root '
            'latency cause. '
            'This result is preparation performance evidence, not a driving latency fix.',
            'Explicit Python seeds must remain in src/scripts or exact root scripts.py. '
            'The old function allowed arbitrary explicit Python paths; that unsafe generality '
            'is intentionally refused, not claimed equivalent.'
        ],
        old_frozen_source_edits=0, discovered_modules_executed=0, model_computed=False,
        device_or_live_calls=0, optimizer_updates=0, fit_allowed=False,
        executed_program_sha256=sha(Path(__file__).read_bytes()))
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    (out/'receipt.json').write_text(json.dumps(dict(
        schema='flybrain.source_closure_inventory.receipt.v1',
        status=result['status'], result_sha256=sha((out/'result.json').read_bytes()),
        executed_program_sha256=result['executed_program_sha256'],
        source_sha256=after, failed_checks=len(failed)), indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], check_count=len(checks),
                         failed_checks=len(failed), timing_seconds=result['timing_seconds'],
                         result_sha256=sha((out/'result.json').read_bytes()))))
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
