import ast
from pathlib import Path

import pytest

from flybrain_drive.source_closure_inventory_v1 import InventoryDriftError, SourceInventory


def fixture_tree(root, files):
    (root/'src').mkdir(exist_ok=True)
    (root/'scripts').mkdir(exist_ok=True)
    for name, text in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf8')
    return SourceInventory(root, files)


@pytest.fixture(scope='module')
def legacy():
    # Execute only the frozen function definition, never its module-level closure
    # or a discovered source module.  Physical legacy stat calls are tiny fixtures.
    root = Path(__file__).parents[1]
    text = (root/'src/flybrain_drive/pedal_train_candidate_host_v4.py').read_text('utf8')
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'local_import_closure')
    namespace = {'ast': ast, 'Path': Path, '_PROJECT_ROOT': root}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen closure only>', 'exec'),
         namespace)
    return namespace['local_import_closure']


def test_package_precedence_and_old_intermediate_package_omission(tmp_path, legacy):
    files = {
        'scripts/start.py': 'import flybrain_drive.pkg.child\nimport flybrain_drive.both\n',
        'src/flybrain_drive/__init__.py': '',
        'src/flybrain_drive/pkg/__init__.py': 'raise RuntimeError("not executed")\n',
        'src/flybrain_drive/pkg/child.py': '',
        'src/flybrain_drive/both.py': '',
        'src/flybrain_drive/both/__init__.py': '',
    }
    inventory = fixture_tree(tmp_path, files)
    result = inventory.local_import_closure(['scripts/start.py'])
    assert result == tuple(sorted(['scripts/start.py', 'src/flybrain_drive/__init__.py',
                                  'src/flybrain_drive/pkg/child.py',
                                  'src/flybrain_drive/both.py']))
    assert result == legacy(['scripts/start.py'], project_root=tmp_path)


def test_relative_imports_and_legacy_over_level_slice(tmp_path, legacy):
    files = {
        'src/flybrain_drive/a/start.py':
            'from . import peer\nfrom .. import shared\nfrom .... import deep\n',
        'src/flybrain_drive/a/__init__.py': '',
        'src/flybrain_drive/a/peer.py': '',
        'src/flybrain_drive/shared.py': '',
        'src/flybrain_drive/deep.py': '',
        'src/flybrain_drive/__init__.py': '',
    }
    inventory = fixture_tree(tmp_path, files)
    result = inventory.local_import_closure(['src/flybrain_drive/a/start.py'])
    assert result == tuple(sorted(files))
    assert result == legacy(['src/flybrain_drive/a/start.py'], project_root=tmp_path)


def test_package_init_relative_wildcard_and_script_root(tmp_path, legacy):
    files = {
        'src/flybrain_drive/__init__.py': 'from . import helper\n',
        'src/flybrain_drive/helper.py': 'from scripts import task\n',
        'scripts/__init__.py': '',
        'scripts/task.py': 'from flybrain_drive import *\n',
    }
    inventory = fixture_tree(tmp_path, files)
    result = inventory.local_import_closure(['src/flybrain_drive/__init__.py'])
    assert result == tuple(sorted(files))
    assert result == legacy(['src/flybrain_drive/__init__.py'], project_root=tmp_path)


def test_optional_repo_root_scripts_module_precedes_package(tmp_path, legacy):
    files = {'scripts/start.py': 'import scripts\n', 'scripts.py': '',
             'scripts/__init__.py': 'raise RuntimeError("wrong module precedence")\n'}
    fixture_tree(tmp_path, files)
    inventory = SourceInventory.capture(tmp_path)
    result = inventory.local_import_closure(['scripts/start.py'])
    assert result == ('scripts.py', 'scripts/start.py')
    assert result == legacy(['scripts/start.py'], project_root=tmp_path)
    assert inventory.verify_unchanged()['source_reads'] == 2


def test_namespace_function_inactive_aliases_and_nonlocal_ignored(tmp_path, legacy):
    files = {
        'scripts/start.py': ('if False:\n import flybrain_drive.ns.leaf as unused\n'
                             'def f():\n from flybrain_drive.ns import other\n'
                             'import torch\nimport outsider\n'),
        'src/flybrain_drive/ns/leaf.py': 'raise RuntimeError("not executed")\n',
        'src/flybrain_drive/ns/other.py': '',
        'src/torch.py': 'raise RuntimeError("not executed")\n',
        'src/outsider.py': '',
    }
    inventory = fixture_tree(tmp_path, files)
    expected = ('scripts/start.py', 'src/flybrain_drive/ns/leaf.py',
                'src/flybrain_drive/ns/other.py')
    assert inventory.local_import_closure(['scripts/start.py']) == expected
    assert expected == legacy(['scripts/start.py'], project_root=tmp_path)


def test_explicit_non_python_bindings_retained_unopened(tmp_path, legacy):
    inventory = fixture_tree(tmp_path, {'scripts/start.py': ''})
    seeds = ['scripts/start.py', 'configs/missing.json', 'C:/external/missing.dll']
    assert inventory.local_import_closure(seeds) == tuple(sorted(seeds))
    assert inventory.local_import_closure(seeds) == legacy(seeds, project_root=tmp_path)
    assert set(inventory.source_sha256) == {'scripts/start.py'}


def test_membership_does_not_query_metadata_and_source_is_read_once(tmp_path, monkeypatch):
    files = {'scripts/start.py': 'import flybrain_drive.leaf\n'*20,
             'src/flybrain_drive/leaf.py': ''}
    inventory = fixture_tree(tmp_path, files)
    reads = []
    original = Path.read_bytes

    def read(path):
        reads.append(str(path))
        return original(path)

    def forbidden(*args, **kwargs):
        raise AssertionError('per-alias metadata query')

    with monkeypatch.context() as context:
        context.setattr(Path, 'is_file', forbidden)
        context.setattr(Path, 'stat', forbidden)
        context.setattr(Path, 'resolve', forbidden)
        context.setattr(Path, 'read_bytes', read)
        assert len(inventory.local_import_closure(['scripts/start.py'])) == 2
        assert len(inventory.local_import_closure(['scripts/start.py'])) == 2
    assert len(reads) == 2


def test_windows_casefold_preserves_legacy_target_spelling(tmp_path):
    files = {'scripts/start.py': 'import flybrain_drive.mixed\n',
             'src/flybrain_drive/Mixed.PY': ''}
    fixture_tree(tmp_path, files)
    insensitive = SourceInventory(tmp_path, files, case_sensitive=False)
    sensitive = SourceInventory(tmp_path, files, case_sensitive=True)
    assert insensitive.local_import_closure(['scripts/start.py']) == (
        'scripts/start.py', 'src/flybrain_drive/mixed.py')
    assert 'src/flybrain_drive/Mixed.PY' in insensitive.source_sha256
    assert sensitive.local_import_closure(['scripts/start.py']) == ('scripts/start.py',)


def test_casefold_collision_rejected(tmp_path):
    with pytest.raises(ValueError, match='Ambiguous'):
        SourceInventory(tmp_path, ['scripts/A.py', 'scripts/a.py'], case_sensitive=False)


@pytest.mark.parametrize('bad', ['../elsewhere.py', '/tmp/a.py', 'configs/a.py',
                                'src/../elsewhere.py', 'C:/outside/a.py'])
def test_python_inventory_cannot_escape_roots(tmp_path, bad):
    with pytest.raises(ValueError, match='src/scripts'):
        SourceInventory(tmp_path, [bad])


def test_scoped_rg_includes_ignored_hidden_and_uppercase_sources(tmp_path):
    files = {'scripts/start.py': 'import flybrain_drive.ignored\n',
             'src/flybrain_drive/ignored.py': '',
             'src/flybrain_drive/.hidden.py': '',
             'src/flybrain_drive/Upper.PY': ''}
    fixture_tree(tmp_path, files)
    (tmp_path/'.gitignore').write_text('src/\n', encoding='utf8')
    (tmp_path/'.ignore').write_text('scripts/\n', encoding='utf8')
    (tmp_path/'outside.py').write_text('raise RuntimeError', encoding='utf8')
    inventory = SourceInventory.capture(tmp_path)
    assert inventory.python_paths == frozenset(files)
    assert inventory.inventory_command[-2:] == ('src', 'scripts')
    assert '--hidden' in inventory.inventory_command
    assert '--no-ignore' in inventory.inventory_command
    assert inventory.local_import_closure(['scripts/start.py']) == (
        'scripts/start.py', 'src/flybrain_drive/ignored.py')
    assert inventory.verify_unchanged()['per_alias_metadata_queries'] == 0
    assert inventory.verified


@pytest.mark.parametrize('change', ['same_size_bytes', 'added_path', 'removed_path'])
def test_seal_rejects_inventory_or_read_drift(tmp_path, change):
    fixture_tree(tmp_path, {'scripts/start.py': 'x=1\n'})
    inventory = SourceInventory.capture(tmp_path)
    inventory.local_import_closure(['scripts/start.py'])
    if change == 'same_size_bytes':
        (tmp_path/'scripts/start.py').write_text('x=2\n', encoding='utf8')
    elif change == 'added_path':
        (tmp_path/'scripts/late.py').write_text('', encoding='utf8')
    else:
        (tmp_path/'scripts/start.py').unlink()
    with pytest.raises(InventoryDriftError):
        inventory.verify_unchanged()
    assert not inventory.verified


def test_missing_seed_is_not_silently_omitted(tmp_path):
    inventory = fixture_tree(tmp_path, {'scripts/start.py': ''})
    with pytest.raises(InventoryDriftError, match='absent'):
        inventory.local_import_closure(['scripts/missing.py'])
