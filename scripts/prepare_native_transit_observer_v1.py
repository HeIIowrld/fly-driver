"""Exact source extraction of validated GET-only observer helpers; no live reads."""

import ast
import hashlib
from pathlib import Path

SOURCE = Path('scripts/transit_to_train_start_v2.py')
SOURCE_SHA256 = '16707542c18a07b82e59c64e6449564dc92bfda72c257575f755e90f25dfa429'
TARGET = Path('src/flybrain_drive/native_transit_observer_v1.py')
FUNCTIONS = ('sha', 'finite', 'check_idle', 'decode_observer', 'validate_pair',
             'stable_native_copy', 'observe_paused')
PREAMBLE = '''"""Derived GET-only native observer; no camera/worker/model/control activation.

The functions are copied byte-exact from the validated1829 source, except the
old807-specific error text now says registered Lua. Neither validate_plan nor
coordinate_link nor the camera-based orchestration is imported or copied.
"""

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

DERIVED_SOURCE_SHA256 = '16707542c18a07b82e59c64e6449564dc92bfda72c257575f755e90f25dfa429'


'''


def build(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Exact validated observer derivation source required')
    text = source.decode('utf8')
    tree = ast.parse(text)
    functions = {n.name: ast.get_source_segment(text, n)
                 for n in tree.body if isinstance(n, ast.FunctionDef)}
    result = PREAMBLE+'\n\n\n'.join(functions[name] for name in FUNCTIONS)+'\n'
    assert result.count('Exact installed807 required; never deploy from staging') == 1
    result = result.replace('Exact installed807 required; never deploy from staging',
                            'Exact registered installed Lua required; never deploy from staging')
    return result.encode()


def main():
    value = build(SOURCE.read_bytes())
    with TARGET.open('xb') as stream:
        stream.write(value)
    print(hashlib.sha256(value).hexdigest())


if __name__ == '__main__':
    main()
