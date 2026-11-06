"""Explicit three-episode cohorts for two-TRAIN/one-development experiments."""
import json
from pathlib import Path

DEFAULT_SPLIT='runs/teacher-synchronized-20261102/student-split-01.json'


def three_episode_cohort(path):
    entries=json.loads(Path(path).read_text(encoding='utf-8'))['episodes']
    if (len(entries)!=3 or any(e['role']!='train' for e in entries[:2])
            or entries[2]['role'] not in ('heldout','development')
            or len({e['episode_id'] for e in entries})!=3
            or len({str(Path(e['data']).resolve()) for e in entries})!=3):
        raise ValueError('Exactly two distinct TRAIN episodes and one distinct development episode required')
    return entries
