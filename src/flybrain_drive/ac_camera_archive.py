"""Portable camera bundles with per-member provenance and verified local replay staging."""

import hashlib
import json
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .evidence import write_json
from .graph import sha256


def checked_name(name):
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or '/' in name or '\\' in name or ':' in name):
        raise ValueError('Camera archive members must have unique flat relative names')
    return name


def bundle_camera_files(sources, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records, files, names = [], [], set()
    for source in sources:
        source = Path(source)
        if not source.exists():
            continue
        for file in sorted(source.iterdir()):
            if not file.is_file() or file.suffix not in (
                    '.json', '.jsonl', '.png', '.bmp', '.lua', '.mmf'):
                continue
            name = checked_name(file.name)
            if name in names:
                raise ValueError('Source archive filename collision')
            names.add(name)
            files.append(file)
    with tempfile.TemporaryDirectory(prefix='flybrain-camera-bundle-') as temporary:
        local = Path(temporary)/'camera.zip'
        with zipfile.ZipFile(local, 'w', compression=zipfile.ZIP_STORED) as bundle:
            for file in files:
                value = file.read_bytes()
                records.append(dict(file=file.name, sha256=hashlib.sha256(value).hexdigest(),
                                    bytes=len(value)))
                bundle.writestr(file.name, value)
        expected = sha256(local)
        target = destination/'camera.zip'
        shutil.copyfile(local, target)
        if sha256(target) != expected:
            raise ValueError('Camera bundle transfer hash mismatch')
        write_json(destination/'archive.json', records)
        write_json(destination/'bundle.json', dict(schema='flybrain.csp.camera_bundle.v1',
            file='camera.zip', sha256=expected, bytes=local.stat().st_size,
            archive_manifest_sha256=sha256(destination/'archive.json')))
    return dict(all_hashes_verified=True, files=len(records), source_retained=True,
                worker_alive=False, format='zip_stored', bundle_sha256=expected)


class CameraArchive:
    """Legacy files or a hash-verified bundle; never extracts archive-supplied paths."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.temporary, self.bundle = None, None
        self.bundle_sha256 = None
        self.staged_directory = None
        manifest = self.directory/'archive.json'
        self.records = json.loads(manifest.read_text(encoding='utf8'))
        self.by_name = {}
        for record in self.records:
            name = checked_name(record['file'])
            if name in self.by_name:
                raise ValueError('Duplicate camera archive member')
            self.by_name[name] = record
        descriptor = self.directory/'bundle.json'
        if descriptor.exists():
            info = json.loads(descriptor.read_text(encoding='utf8'))
            if (info.get('schema') != 'flybrain.csp.camera_bundle.v1'
                    or info['file'] != 'camera.zip'
                    or sha256(manifest) != info['archive_manifest_sha256']):
                raise ValueError('Camera bundle descriptor or manifest differs')
            self.temporary = tempfile.TemporaryDirectory(prefix='flybrain-camera-replay-')
            local = Path(self.temporary.name)/'camera.zip'
            shutil.copyfile(self.directory/'camera.zip', local)
            if local.stat().st_size != info['bytes'] or sha256(local) != info['sha256']:
                raise ValueError('Camera replay staging hash differs')
            self.bundle = zipfile.ZipFile(local)
            self.bundle_sha256 = info['sha256']
            members = self.bundle.namelist()
            if len(set(members)) != len(members) or set(members) != set(self.by_name):
                raise ValueError('Camera bundle members differ from the manifest')
        else:
            self.temporary = tempfile.TemporaryDirectory(prefix='flybrain-camera-files-replay-')
            self.staged_directory = Path(self.temporary.name)

    def read(self, name):
        name = checked_name(name)
        if name not in self.by_name:
            raise ValueError('Camera file absent from verified manifest')
        if self.bundle:
            value = self.bundle.read(name)
        else:
            path = self.staged_directory/name
            if not path.exists():
                path = (self.directory/name).resolve()
                if not path.is_relative_to(self.directory.resolve()):
                    raise ValueError('Camera file escaped the archive directory')
            value = path.read_bytes()
        record = self.by_name[name]
        if len(value) != record['bytes'] or hashlib.sha256(value).hexdigest() != record['sha256']:
            raise ValueError('Camera archive member hash differs')
        if self.staged_directory is not None and path.parent != self.staged_directory:
            (self.staged_directory/name).write_bytes(value)
        return value

    def json(self, name):
        return json.loads(self.read(name))

    def verify_all(self):
        if self.bundle is not None:
            for name in self.by_name:
                self.read(name)
        else:
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix='camera-replay') as pool:
                # Each name occurs once; independent verified files share no write target.
                for _ in pool.map(self.read, self.by_name):
                    pass

    def close(self):
        if getattr(self, 'bundle', None) is not None:
            self.bundle.close()
            self.bundle = None
        if getattr(self, 'temporary', None) is not None:
            self.temporary.cleanup()
            self.temporary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
