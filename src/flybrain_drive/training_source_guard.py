"""Fail-closed admission for real recordings used to create training datasets.

Raw replay success and a teacher-active flag do not grant fit permission.
"""

def require_training_source(*, recording, raw_audit, label_audit):
    if not all(isinstance(item,dict) for item in (recording,raw_audit,label_audit)):
        raise ValueError('Explicit recording, raw and label admissions required')
    if recording.get('training_admissible') is not True:
        raise ValueError('Recording is not explicitly admitted for training')
    if raw_audit.get('continuous_40ms_training_admissible') is not True:
        raise ValueError('Raw camera/body timing is not explicitly admitted for training')
    if label_audit.get('training_admissible') is not True:
        raise ValueError('Teacher labels are not explicitly admitted for training')
