"""I-F10  Estimator-calibrated seed dynamics (register #26, #27; no commitment wording).
(a) Raw sigma50 vs ceiling-free relative onset as the number of generations n grows (archive SD3.5 M, uncentred; R5 H1).
(b) Matched n = 60 prompts, 50 subsets: sigma50 vs relative onset per channel, median and 5-95% (R6 S3a-c).
(c) VR(4) on the calibrated scale: observed / simulated random walk, with the AR(0.8) and iid-jitter references (R6 S3d).
Sources: review/round5/out/H1_seedfate_vs_sample_size.csv; review/round6/out/S3_estimators.csv
"""
from common_I import *

H1 = pd.read_csv(f'{R5}/H1_seedfate_vs_sample_size.csv')
H1 = H1[H1.set.str.startswith('archive SD35M')].copy()
S3 = pd.read_csv(f'{R6}/S3_estimators.csv')
rows = []
for r in H1.itertuples():
    rows.append(dict(panel='a', set=r.set, prompts=r.prompts, runs_per_prompt=r.runs_per_prompt, n=r.n, sigma50=r.sigma50, sigma50_p5=r.sigma50_p5,
                     sigma50_p95=r.sigma50_p95, sigma50_undefined_frac=r.sigma50_undefined_frac, rel_onset=r.rel_onset, rel_p5=r.rel_p5, rel_p95=r.rel_p95,
                     source='round5/out/H1_seedfate_vs_sample_size.csv'))
for r in S3.itertuples():
    rows.append(dict(panel='b', set=r.channel, n=60, sigma50=r.matched60_s50_median, sigma50_p5=r.matched60_s50_p5, sigma50_p95=r.matched60_s50_p95,
                     rel_onset=r.matched60_rel_median, rel_p5=r.matched60_rel_p5, rel_p95=r.matched60_rel_p95, source='round6/out/S3_estimators.csv'))
    rows.append(dict(panel='c', set=r.channel, VR4=r.VR4, VR4_walk=r.VR4_walk, VR4_ar08=r.VR4_ar08, VR4_iid=r.VR4_iid, VR4_over_walk=r.VR4_over_walk,
                     ar08_over_walk=r.VR4_ar08 / r.VR4_walk, iid_over_walk=r.VR4_iid / r.VR4_walk, source='round6/out/S3_estimators.csv'))
D = pd.DataFrame(rows)
write_csv(D, 'I-F10_estimators')


def chlab(c):
    m, s = c.split('-'); return f"{SHORT[m]} {'MLP' if s == 'mlp' else 'attn'}"


fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.6), gridspec_kw=dict(width_ratios=[0.95, 1.2, 0.9], wspace=0.5))
ax = axs[0]
a = D[D.panel == 'a'].copy()
a['lab'] = a.prompts.astype(int).astype(str) + '×' + a.runs_per_prompt.astype(int).astype(str)
for k, col, mk, lb in [('sigma50', C[1], 'o', 'raw σ50'), ('rel_onset', C[0], 'D', 'relative onset')]:
    d = a[a[k.replace('rel_onset', 'rel_onset')].notna()]
    ok = d.sigma50_undefined_frac < 0.5 if k == 'sigma50' else np.ones(len(d), bool)
    lo = d[k.replace('sigma50', 'sigma50_p5').replace('rel_onset', 'rel_p5')]; hi = d[k.replace('sigma50', 'sigma50_p95').replace('rel_onset', 'rel_p95')]
    jit = 1.04 if k == 'rel_onset' else 0.96
    ax.errorbar(d.n[ok] * jit, d[k][ok], yerr=[(d[k] - lo)[ok], (hi - d[k])[ok]], fmt=mk, ms=3.5, color=col, lw=0.9, capsize=1.5, label=lb)
    if k == 'sigma50' and (~ok).any():
        dd = d[~ok]
        ax.scatter(dd.n * jit, dd[k], marker=mk, s=14, facecolor='white', edgecolor=col, lw=0.9, zorder=3)
        ax.annotate('σ50 undefined in\n98% of folds', (dd.n.iloc[0] * jit, dd[k].iloc[0]), xytext=(7, -3), textcoords='offset points', fontsize=6.3, color=INK2)
ax.set_xscale('log'); ax.set_xticks([60, 120, 240, 480], ['60', '120', '240', '480']); ax.minorticks_off()
ax.set_xlabel('generations n'); ax.set_ylabel('σ at onset')
ax.set_ylim(0, 1.05)
ax.legend(loc='center right', fontsize=6.6, handletextpad=0.3)
panel_label(ax, 'a')

ax = axs[1]
b = D[D.panel == 'b'].reset_index(drop=True); y = np.arange(len(b))
ax.errorbar(b.sigma50, y - 0.15, xerr=[b.sigma50 - b.sigma50_p5, b.sigma50_p95 - b.sigma50], fmt='o', ms=3.5, color=C[1], lw=0.9, capsize=1.5, label='raw σ50')
ax.errorbar(b.rel_onset, y + 0.15, xerr=[b.rel_onset - b.rel_p5, b.rel_p95 - b.rel_onset], fmt='D', ms=3.2, color=C[0], lw=0.9, capsize=1.5, label='relative onset')
ax.set_yticks(y, [chlab(c) for c in b.set], fontsize=7); ax.set_ylim(len(b) - 0.4, -1.0)
ax.set_xlabel('σ at onset (n = 60)'); ax.set_xlim(0.3, 1.0)
ax.legend(loc='lower left', fontsize=6.6, handletextpad=0.3)
panel_label(ax, 'b')

ax = axs[2]
c = D[D.panel == 'c'].reset_index(drop=True)
ax.axvline(1.0, color=INK2, lw=0.8); ax.axvline(c.ar08_over_walk.mean(), color=INK2, lw=0.8, ls='--'); ax.axvline(c.iid_over_walk.mean(), color=INK2, lw=0.8, ls=':')
for xv, t, ha in [(1.0, ' walk', 'left'), (c.ar08_over_walk.mean(), 'AR(0.8) ', 'right'), (c.iid_over_walk.mean(), 'iid ', 'right')]:
    ax.text(xv, -0.85, t, rotation=0, fontsize=6.5, color=INK2, ha=ha, va='bottom')
ax.scatter(c.VR4_over_walk, y, s=22, color=INK, zorder=3, lw=0)
for j, v in enumerate(c.VR4_over_walk):
    ax.text(v + 0.04, j, f'{v:.2f}', va='center', fontsize=6.5, color=INK2)
ax.set_yticks(y, [''] * len(c)); ax.set_ylim(len(c) - 0.4, -1.0)
ax.set_xlim(0.15, 1.35); ax.set_xlabel('VR(4) / simulated walk')
panel_label(ax, 'c')
save(fig, 'I-F10_estimators')
print('I-F10 done')
