"""Shared paths and helpers for the Mode-I figure scripts.

Every figure script imports this module, builds one figure from recorded outputs (or, where noted, recomputes from the
raw telemetry cubes with the Round 2-6 estimators unchanged), writes data/<figure>.csv with the exact values plotted and
calls figstyle.save().  Nothing here changes any number; it only locates inputs.
"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common'))  # <repo root>/common/figstyle.py
from figstyle import *          # noqa: F401,F403  (C, GRAY, LIGHT, INK, INK2, MODEL_COLOR, MODEL_LABEL, TEXTWIDTH, save, panel_label, plt)

PAPER = f'{_COSPAFS_ROOT}/papers/mode-I'
DATA = os.path.join(PAPER, 'data')
REVIEW = os.environ.get('REVIEW', f'{_COSPAFS_ROOT}/review')  # folder holding the review rounds' outputs
R2 = f'{REVIEW}/round2/out'
R4 = f'{REVIEW}/round4/out'
R5 = f'{REVIEW}/round5/out'
R6 = f'{REVIEW}/round6/out'
LEDGER = os.environ.get('LEDGER', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '_sources', 'ledger_v020_text.txt'))
# Scratch folder that holds the raw cubes (F_*.pkl, R5_*.pkl, F3_*.pkl), the parquet archive (pq/) and the v11 figure-data
# package (fig/data/). Override with the SCR environment variable, exactly as round5/scripts/r5_common.py does.
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
V11 = os.path.join(SCR, 'fig', 'data')
os.makedirs(DATA, exist_ok=True)

MODELS = ['ZIT', 'ZIB', 'MOCHI', 'SD35M', 'SD35L']
SHORT = {'ZIT': 'ZIT', 'ZIB': 'ZIT-25', 'MOCHI': 'Mochi', 'SD35M': 'SD3.5 M', 'SD35L': 'SD3.5 L'}
SURF = 'white'


def write_csv(df, name):
    """Write the exact plotted values to data/<name>.csv and return the path."""
    p = os.path.join(DATA, name + '.csv')
    df.to_csv(p, index=False)
    return p


def rjson(path):
    return json.load(open(path))


def cubes():
    """Import the Round 5/6 cube loaders (ARCH, R5, R3, prof, tangent, ...) with SCR pointing at the scratch folder."""
    os.environ.setdefault('SCR', SCR)
    sys.path.insert(0, f'{REVIEW}/round6/scripts')
    import r6c_common as rc   # noqa: E402  (also exposes r5_common names)
    return rc


def pair_label(a, b):
    return f'{SHORT[a]}–{SHORT[b]}'


def ledger_has(*snippets):
    """Assert that each transcribed snippet occurs verbatim in the ledger text (guards transcription errors)."""
    if not os.path.exists(LEDGER):  # the findings ledger is internal (available on request); skip the transcription check
        return True
    txt = open(LEDGER, encoding='utf-8').read()
    for s in snippets:
        if s not in txt:
            raise ValueError(f'ledger snippet not found: {s!r}')
    return True
