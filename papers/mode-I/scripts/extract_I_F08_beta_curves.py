"""Extract the binned beta_layer(g) curves for I-F8 panel (c) with the CoS-PAFS Live Analyzer v3, unchanged.

For each channel the archive CSVs are loaded through the analyzer's own DataManager.load_paths (as the dashboard's
Load Paths button does), filtered to the submodule as in review/round6/scripts/r6c_analyzer_beta_nulls.py, and passed
to the analyzer's plot_beta_layer. The curve is read back from the figure the analyzer draws:
  - binned mean beta: the line plotted with lw=2 (29 bins over the g range, bins with >= 10 points);
  - +/- 1 s.d. band: the fill_between polygon;
  - crossings g* and their direction: parsed from the figure title.
Nothing is recomputed outside the analyzer. Needs the raw telemetry: set TEL=<folder with the per-campaign CSV folders>.
Output: data/I-F8c_beta_curves.csv (curves) and data/I-F8c_beta_crossings.csv.
"""
import os as _os; _COSPAFS_ROOT = _os.environ.get('COSPAFS_ROOT', _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))); _COSPAFS_CACHE = _os.environ.get('COSPAFS_CACHE', _os.path.join(_COSPAFS_ROOT, 'cache'))  # repo-relative paths (release)
import os, sys, glob, re, types, importlib.util
sys.modules['gradio'] = types.ModuleType('gradio')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.environ.get('REVIEW', f'{_COSPAFS_ROOT}/review')
ANALYZER = os.environ.get('ANALYZER', f'{REVIEW}/round6/analyzer/cos_pafs_live_analyzer.py')
SCR = os.environ.get('SCR', f'{_COSPAFS_CACHE}')
TEL = os.environ.get('TEL', f'{SCR}/tel/DM')
DATA = os.path.join(HERE, '..', 'data')

spec = importlib.util.spec_from_file_location('cpla', ANALYZER); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

# same channels, folders and submodule filters as the Round 6 A4 replay (r6c_analyzer_beta_nulls.py)
CH = [('SD3.5 M attn', 'SD3.5MED-CFG-4.5-attn-1024-3TE-120', None),
      ('SD3.5 L attn', 'SD3.5LAR-CFG-3.5-attn-1024-3TE-120', None),
      ('ZIB attn', 'ZIB-CFG-4.0-attn-1024-120', 'attn'),
      ('ZIB MLP', 'ZIB-CFG-4.0-attn-1024-120', 'mlp'),
      ('ZIT MLP', 'ZIT-CFG-1.0-attn-1024-120', 'mlp'),
      ('Mochi attn', 'MOCHI-CFG-4.5-attn-480-120', None),
      ('ZIT attn', 'ZIT-CFG-1.0-attn-1024-120', 'attn')]

curves, cross = [], []
for lab, folder, sub in CH:
    dm = A.DataManager(); dm.load_paths(','.join(sorted(glob.glob(f'{TEL}/{folder}/*.csv'))))
    df = dm.get_df(); df = df[df.g_micro.notna()]
    if sub: df = df[df.submodule == sub]
    fig = A.plot_beta_layer(df); ax = fig.axes[0]
    line = [l for l in ax.get_lines() if l.get_linewidth() == 2][0]
    x, m = (np.asarray(v, float) for v in line.get_data())
    band = ax.collections[-1].get_paths()[0].vertices          # fill_between polygon: lower edge forward, upper edge back
    lo = np.interp(x, *band[:len(band) // 2].T[[0, 1]][:, np.argsort(band[:len(band) // 2, 0])])
    hi = np.interp(x, *band[len(band) // 2:].T[[0, 1]][:, np.argsort(band[len(band) // 2:, 0])])
    for xi, mi, l_, h_ in zip(x, m, lo, hi):
        curves.append(dict(channel=lab, g_center=xi, beta_mean=mi, beta_lo_1sd=min(l_, h_), beta_hi_1sd=max(l_, h_),
                           gens=df.gen_idx.nunique(), source='analyzer plot_beta_layer (unchanged), archive telemetry'))
    for gs, d in re.findall(r'g\*=([0-9.]+) \(([a-z]+→[a-z]+)\)', ax.get_title()):
        cross.append(dict(channel=lab, g_star=float(gs), direction=d))
    plt.close(fig)
    print(lab, len(x), 'bins;', ' '.join(f"{c['g_star']:.4f}{'+-' if c['direction']=='pos→neg' else '-+'}" for c in cross if c['channel'] == lab), flush=True)

pd.DataFrame(curves).to_csv(os.path.join(DATA, 'I-F8c_beta_curves.csv'), index=False)
pd.DataFrame(cross).to_csv(os.path.join(DATA, 'I-F8c_beta_crossings.csv'), index=False)
