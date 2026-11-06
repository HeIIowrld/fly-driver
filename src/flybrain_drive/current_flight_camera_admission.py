"""Actual control always requires normal camera admission.

A stopped/failed metadata conflict is handled by the separate stationary-only
reconciler; this module never turns that diagnostic allowance into driving.
"""
from .paused_mmf_startup import PausedMMFStartupGate


def admit(status,previous,*,camera_root,status_path,pin,completed_evidence=None):
    return status,PausedMMFStartupGate(status,*previous),None
