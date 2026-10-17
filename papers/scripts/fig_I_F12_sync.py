"""I-F12  Synchronization falsification (register #5: DO NOT USE as an event).
(a) R(sigma) of the population-mean field for two channels, observed vs the per-layer straight-line surrogate
    (recomputed here with round2/scripts/common.syncR and the S5 trend() definition, unchanged).
(b) Observed R peak vs the monotone-warp null (median to p95, 500 draws) and the straight-line surrogate, 7 channels (R6 S5).
Sources: review/round6/out/S5_sync.csv; raw cubes F_*.pkl (panel a only); review/round6/scripts/r6c_S5_sync.py
"""
from common_I import *

rc = cubes()
sys.path.insert(0, f'{REVIEW}/round2/scripts')
from common import syncR  # noqa: E402


def trend(M):  # identical to r6c_S5_sync.trend
    T = M.shape[0]; t = np.arange(T); A = np.stack([np.ones(T), t], 1); out = np.empty_like(M)
    for l in range(M.shape[1]):
        for v in range(3):
            c, *_ = np.linalg.lstsq(A, M[:, l, v], rcond=None); out[:, l, v] = A @ c
    return out


S5 = pd.read_csv(f'{R6}/S5_sync.csv')
rows = []
CH_A = [('ZIB', 'attn'), ('SD35M', 'attn')]
for m, sub in CH_A:
    g, meta, X, s = rc.arch(m, sub); M = X.mean(0)
    Ro = syncR(M); Rt = syncR(trend(M))
    ref = S5.set_index('channel').loc[f'{m}-{sub}']
    assert abs(Ro.max() - ref.R_peak) < 1e-9 and abs(Rt.max() - ref.trend_R_peak) < 1e-9
    for t in range(len(s)):
        rows.append(dict(panel='a', channel=f'{m}-{sub}', sigma=s[t], step=t, R_observed=Ro[t], R_straight_line=Rt[t]))
for r in S5.itertuples():
    rows.append(dict(panel='b', channel=r.channel, R_peak=r.R_peak, peak_sigma=r.peak_sigma, warp_null_median=r.warp_null_median,
                     warp_null_p95=r.warp_null_p95, warp_p=r.warp_p, trend_R_peak=r.trend_R_peak))
D = pd.DataFrame(rows); D['source'] = np.where(D.panel == 'a', 'recomputed from F_*.pkl (syncR, S5 trend)', 'round6/out/S5_sync.csv')
write_csv(D, 'I-F12_sync_falsification')

fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.55), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.32))
ax = axs[0]
for (m, sub), col in zip(CH_A, [MODEL_COLOR['ZIB'], MODEL_COLOR['SD35M']]):
    d = D[(D.panel == 'a') & (D.channel == f'{m}-{sub}')]
    ax.plot(d.sigma, d.R_observed, color=col, lw=1.4)
    ax.plot(d.sigma, d.R_straight_line, color=col, lw=1.0, ls='--')
    xs, dy = {'ZIB': (0.62, -0.55), 'SD35M': (0.42, 0.18)}[m]
    i = int(np.argmin(np.abs(d.sigma.values - xs)))
    ax.text(d.sigma.iloc[i], d.R_observed.iloc[i] + dy, SHORT[m] + ' attn', color=INK, fontsize=7, ha='left', va='center')
ax.plot([], [], color=INK2, lw=1.4, label='observed'); ax.plot([], [], color=INK2, lw=1.0, ls='--', label='straight line per layer')
ax.invert_xaxis(); ax.set_xlabel('σ'); ax.set_ylabel('Synchronization R(σ)')
ax.set_ylim(None, 1.12)
ax.legend(loc='lower center', fontsize=6.6)
panel_label(ax, 'a')

ax = axs[1]
lab = [c.replace('-attn', ' attn').replace('-mlp', ' MLP').replace('MOCHI', 'Mochi').replace('SD35M', 'SD3.5 M').replace('SD35L', 'SD3.5 L') for c in S5.channel]
y = np.arange(len(S5))
ax.hlines(y, S5.warp_null_median, S5.warp_null_p95, color=LIGHT, lw=5, label='warp null, median–p95')
ax.scatter(S5.R_peak, y, s=22, color=INK, zorder=3, lw=0, label='observed peak')
ax.scatter(S5.trend_R_peak, y, s=20, marker='D', facecolor='white', edgecolor=INK, zorder=3, lw=0.9, label='straight-line surrogate')
for j, r in enumerate(S5.itertuples()):
    ax.text(1.012, j, f'p={r.warp_p:.3f}', va='center', fontsize=6.6, color=INK2)
ax.set_yticks(y, lab, fontsize=7); ax.invert_yaxis()
ax.set_xlim(0.72, 1.06); ax.set_xticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_xlabel('Peak of R(σ)')
ax.legend(loc='upper center', fontsize=6.6, bbox_to_anchor=(0.45, -0.2), ncol=3, handletextpad=0.3, columnspacing=1.0)
ax.text(1.0, 1.04, 'FALSIFIED as a mid-run event', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5, color=INK, fontweight='bold')
panel_label(ax, 'b')
save(fig, 'I-F12_sync_falsification')
print('I-F12 done')
