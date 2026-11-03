"""AST-only local closure from one scoped path inventory, without module imports.

This is preparation code, not a replacement for a frozen capture contract.  A
caller must compare its first use with its sealed legacy closure, and call
``verify_unchanged`` before sealing a result.  Inventory and read rechecks detect
observed drift, not a filesystem transaction or a change which was later undone.

rg's normal non-following traversal is retained.  Equivalence is scoped to the
captured regular-file namespace; this module makes no universal symlink/junction
equivalence claim.  An omitted alias must fail the independent legacy comparison.
"""

import ast
import hashlib
import os
import subprocess
from pathlib import Path, PurePosixPath


class InventoryDriftError(ValueError):
    """The scoped path inventory or previously read source bytes changed."""


def _local_python_path(value):
    value = value.replace('\\', '/')
    path = PurePosixPath(value)
    if value == 'scripts.py':
        return value  # The old scripts module-before-package probe is repo-root.
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or path.parts[0] not in ('src', 'scripts') or ':' in value):
        raise ValueError('Python paths must stay inside explicit src/scripts roots: '+value)
    return path.as_posix()


def _inventory(project_root, rg_executable):
    # No shell, glob expansion, ignore filtering, package import, or root search.
    command = [rg_executable, '--files', '--hidden', '--no-ignore',
               '--iglob', '*.py', 'src', 'scripts']
    completed = subprocess.run(command, cwd=project_root, check=False,
                               capture_output=True, encoding='utf-8', errors='strict')
    if completed.returncode not in (0, 1) or completed.stderr.strip():
        raise ValueError('Scoped rg inventory failed: '+completed.stderr.strip())
    paths = [_local_python_path(line) for line in completed.stdout.splitlines()]
    if any(not p.lower().endswith('.py') for p in paths) or len(paths) != len(set(paths)):
        raise ValueError('Malformed or duplicate scoped Python inventory')
    # One exact legacy candidate, not a root traversal or a per-alias stat.
    # Path.is_file used to try this before scripts/__init__.py on every alias.
    try:
        root_script = (Path(project_root)/'scripts.py').read_bytes()
    except FileNotFoundError:
        root_script = None
    if root_script is not None:
        paths.append('scripts.py')
    return frozenset(paths), tuple(command), root_script


class SourceInventory:
    """Resolve legacy flybrain_drive/scripts imports by set membership only.

    Non-Python entries are returned verbatim and never opened.  Python seed reads
    are restricted to src/scripts and the exact legacy root scripts.py probe,
    including explicit seeds not discoverable by
    ordinary imports.  No Path.is_file/stat/resolve call occurs during resolution.
    Windows lookup is case-insensitive, but the returned target keeps the exact
    legacy lexical spelling.  Pass case_sensitive explicitly only in fixtures.
    """

    def __init__(self, project_root, python_paths, *, case_sensitive=None,
                 inventory_command=(), rg_executable='rg'):
        self.project_root = Path(project_root)
        self.python_paths = frozenset(_local_python_path(p) for p in python_paths)
        self.case_sensitive = os.name != 'nt' if case_sensitive is None else case_sensitive
        self.inventory_command = tuple(inventory_command)
        self.rg_executable = rg_executable
        self._paths = {}
        for name in self.python_paths:
            key = self._key(name)
            if key in self._paths and self._paths[key] != name:
                raise ValueError('Ambiguous case-folded Python inventory: '+name)
            self._paths[key] = name
        self._reads = {}
        self._preloaded = {}
        self._closures = []
        self.verified = False

    @classmethod
    def capture(cls, project_root, *, rg_executable='rg'):
        """Run exactly one scoped inventory; no discovered source is read yet."""
        paths, command, root_script = _inventory(project_root, rg_executable)
        result = cls(project_root, paths, inventory_command=command,
                     rg_executable=rg_executable)
        if root_script is not None:
            result._preloaded['scripts.py'] = root_script
        return result

    def _key(self, value):
        return value if self.case_sensitive else value.casefold()

    def _actual(self, value):
        return self._paths.get(self._key(_local_python_path(value)))

    def resolve(self, module):
        """Exact old module.py then package/__init__.py, for the two old roots."""
        parts = module.split('.')
        if parts[0] not in ('flybrain_drive', 'scripts'):
            return None
        prefix = PurePosixPath('src') if parts[0] == 'flybrain_drive' else PurePosixPath()
        target = prefix.joinpath(*parts).with_suffix('.py').as_posix()
        if self._actual(target) is None:
            target = prefix.joinpath(*parts, '__init__.py').as_posix()
        return target if self._actual(target) is not None else None

    def _source(self, name):
        actual = self._actual(name)
        if actual is None:
            raise InventoryDriftError('Python seed absent from scoped inventory: '+name)
        if actual not in self._reads:
            try:
                data = (self._preloaded.pop(actual) if actual in self._preloaded
                        else (self.project_root/actual).read_bytes())
            except OSError as exc:
                raise InventoryDriftError('Inventoried source is unavailable: '+actual) from exc
            # Match legacy read_text(encoding='utf8'); a BOM is not silently stripped.
            tree = ast.parse(data.decode('utf8'), filename=name)
            self._reads[actual] = (hashlib.sha256(data).hexdigest(), len(data), tree)
        return self._reads[actual][2]

    def local_import_closure(self, entries):
        """Return a provisional sorted closure; seal only after verify_unchanged.

        ast.walk includes function-local and inactive imports.  The intentionally
        conservative old alias and top-level-package fallbacks, relative slicing,
        wildcard behavior, and omission of unmentioned intermediate packages are
        preserved.  Dynamic imports are not interpreted, as in the old resolver.
        """
        entries = tuple(entries)
        if any(not isinstance(p, str) for p in entries):
            raise ValueError('Explicit source seed strings required')
        sources = set(entries)
        pending = [p for p in entries if p.endswith('.py')]
        self.verified = False
        while pending:
            name = pending.pop()
            path = PurePosixPath(_local_python_path(name))
            parts = list(path.with_suffix('').parts)
            if parts[0] == 'src':
                parts = parts[1:]
            if parts[-1] == '__init__':
                parts = parts[:-1]
                package = parts
            else:
                package = parts[:-1]
            for node in ast.walk(self._source(name)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = package[:len(package)-node.level+1] if node.level else []
                    module = ('.'.join(base+node.module.split('.')) if node.module
                              else '.'.join(base))
                    modules = [module, *(module+'.'+alias.name for alias in node.names)]
                for module in modules:
                    for candidate in (module, module.split('.')[0]):
                        target = self.resolve(candidate)
                        if target and target not in sources:
                            sources.add(target)
                            pending.append(target)
        result = tuple(sorted(sources))
        self._closures.append(result)
        return result

    @property
    def source_sha256(self):
        """Hashes of actual reachable Python paths; non-Python bindings stay external."""
        return {name: value[0] for name, value in sorted(self._reads.items())}

    def verify_unchanged(self):
        """One sealing reinventory and one reread per source; fail on observed drift.

        This is an explicit second inventory, never a per-alias metadata query.
        Call after all closures sharing this inventory have been computed.
        """
        self.verified = False
        paths, _, root_script = _inventory(self.project_root, self.rg_executable)
        if paths != self.python_paths:
            raise InventoryDriftError('Scoped Python inventory changed: added='+
                                      repr(sorted(paths-self.python_paths))+' removed='+
                                      repr(sorted(self.python_paths-paths)))
        for name, (expected, size, _) in sorted(self._reads.items()):
            try:
                data = (root_script if name == 'scripts.py'
                        else (self.project_root/name).read_bytes())
            except OSError as exc:
                raise InventoryDriftError('Previously read source is unavailable: '+name) from exc
            if len(data) != size or hashlib.sha256(data).hexdigest() != expected:
                raise InventoryDriftError('Previously read source bytes changed: '+name)
        self.verified = True
        return dict(status='passed_scoped_inventory_and_source_recheck',
                    inventory_python_paths=len(paths), source_reads=len(self._reads),
                    closures=len(self._closures), inventory_calls_per_capture_and_seal=2,
                    per_alias_metadata_queries=0, filesystem_transaction_claimed=False,
                    discovered_modules_executed=0,
                    namespace_scope='rg non-followed captured paths; legacy equivalence required')
