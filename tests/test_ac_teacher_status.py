"""AI identity is collection metadata, never a fly actor observation field."""

import ast
from pathlib import Path

from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX


def test_teacher_flag_offset_matches_pinned_physics_layout_without_executing_source():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'assets/external/ac-native-memory-20261027/upstream/sim_info.py')
                     .read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'SPageFilePhysics')
    fields = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == '_fields_' for t in n.targets))
    offset = 0
    for item in fields.elts:
        name, kind = item.elts
        count = 1
        if isinstance(kind, ast.BinOp):
            kind, count = kind.left, ast.literal_eval(kind.right)
        assert kind.id in ('c_int32', 'c_float')
        if ast.literal_eval(name) == 'isAIControlled':
            assert offset == 416
            break
        offset += count * 4
    else:
        raise AssertionError('Missing teacher flag')
    assert AI_STATUS_PREFIX.size == 420
