from flybrain_drive.evidence import source_hashes
from flybrain_drive.graph import sha256


def test_parallel_source_hashes_preserve_exact_files_order_and_bytes(tmp_path):
    included = ('src/a.py', 'src/nested/b.py', 'configs/c.json', 'scripts/d.py',
                'integrations/ac/e.py', 'integrations/ac_csp/f.lua')
    excluded = ('src/__pycache__/cache.py', 'src/notes.txt', 'runs/output.py', '.env')
    for relative in included+excluded:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    actual = source_hashes(tmp_path)
    assert list(actual) == list(included)
    assert actual == {name: sha256(tmp_path/name) for name in included}
