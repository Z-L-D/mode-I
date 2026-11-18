"""Rebuild every Mode-I figure (main I-F1..I-F15, supplement I-S1..I-S5).

Usage:  python make_all_figures.py [--recompute] [--only I-F5 I-S2 ...]
  --recompute   also redo the cached I-F5 decoding sweep (about 15 min on 4 cores); otherwise I-F5 reads
                data/I-F5_semantic_decoding.csv.
  --only        build only the listed figure ids.
Each script writes figures/<name>.pdf/.png (via figstyle.save) and data/<name>.csv with the plotted values.
Raw-cube inputs are read from $SCR (default: the session scratchpad; see common_I.py).
"""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [('I-F1', 'fig_I_F01_corpus.py'), ('I-F2', 'fig_I_F02_backbone.py'), ('I-F3', 'fig_I_F03_hierarchy.py'),
           ('I-F4', 'fig_I_F04_capture_regime.py'), ('I-F5', 'fig_I_F05_semantic.py'), ('I-F6', 'fig_I_F06_rsa.py'),
           ('I-F7', 'fig_I_F07_sigma_topology.py'), ('I-F8', 'fig_I_F08_layer_beta.py'), ('I-F9', 'fig_I_F09_state_vs_dynamics.py'),
           ('I-F10', 'fig_I_F10_estimators.py'), ('I-F11', 'fig_I_F11_interventions.py'), ('I-F12', 'fig_I_F12_sync.py'),
           ('I-F13', 'fig_I_F13_unet.py'), ('I-F14', 'fig_I_F14_ace.py'), ('I-F15', 'fig_I_F15_tiers.py'), ('I-S1', 'fig_I_S01_theme_residual.py'), ('I-S2', 'fig_I_S02_stepcount.py'), ('I-S3', 'fig_I_S03_qwen_context.py'),
           ('I-S4', 'fig_I_S04_negative_atlas.py'), ('I-S5', 'fig_I_S05_domain_confusion.py')]


def main(argv):
    recompute = '--recompute' in argv
    only = set(argv[argv.index('--only') + 1:]) if '--only' in argv else None
    failed = []
    for fid, script in SCRIPTS:
        if only and fid not in only:
            continue
        cmd = [sys.executable, os.path.join(HERE, script)] + (['--recompute'] if recompute and fid == 'I-F5' else [])
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        status = 'ok' if r.returncode == 0 else f'FAILED (exit {r.returncode})'
        print(f'{fid:6s} {script:34s} {status}  {time.time() - t0:5.1f}s', flush=True)
        if r.returncode != 0:
            failed.append(fid)
            print(r.stderr[-3000:], file=sys.stderr)
    if failed:
        print('failed:', ' '.join(failed)); return 1
    print('all figures built')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
