"""I-F13  Out-of-family check: SD1.5 U-Net (R3 Part C, 2026-10-08).
(a) C1 (preregistered): per-block g / g_MP (median over gens, cond), range over the 50 steps; DiT archive median ratios
    as a band; the 0.10 prediction threshold.
(b) C2 / C3 (preregistered): cross-validated Delta R^2 of a block effect over a spline in g (C2) and over g + U-Net level
    on within-level pairs (C3), with the DiT archive range for C2 (lens round R1, 0.49-0.65) and the prediction thresholds.
(c) P6 (post hoc): Mantel rho between SD1.5 and each DiT on the 116 shared prompts at the first, middle and last step,
    against the unrelated DiT pairs recomputed on the same prompts.
Sources: review/r3_unet/out/C1_cells.csv, C_scores.csv, P6_rsa.csv; review/lenses/out/PH_isotropy_baseline.csv, L2_R1_autonomy.csv
"""
from common_I import *

U = f'{REVIEW}/r3_unet/out'; LZ = f'{REVIEW}/lenses/out'
cells = pd.read_csv(f'{U}/C1_cells.csv'); S = pd.read_csv(f'{U}/C_scores.csv'); S = S[S.branch == 'cond'].set_index('channel')
rsa = pd.read_csv(f'{U}/P6_rsa.csv'); iso = pd.read_csv(f'{LZ}/PH_isotropy_baseline.csv'); r1 = pd.read_csv(f'{LZ}/L2_R1_autonomy.csv')
dit_ratio = (iso.ratio_median_to_iso.min(), iso.ratio_median_to_iso.max()); dit_dr2 = (r1.dR2.min(), r1.dR2.max())

agg = cells.groupby(['channel', 'block', 'N', 'D']).ratio.agg(['median', 'min', 'max']).reset_index()
unrel = rsa[rsa.a.isin(MODELS) & rsa.b.isin(MODELS) & ~((rsa.a == 'ZIT') & (rsa.b == 'ZIB')) & ~((rsa.a == 'SD35M') & (rsa.b == 'SD35L'))]
sdp = rsa[rsa.a.isin(['SD15attn', 'SD15mlp']) & rsa.b.isin(MODELS)]
out = pd.concat([agg.assign(panel='a', source='r3_unet/out/C1_cells.csv'),
                 S.reset_index()[['channel', 'C2_dR2', 'C3_dR2_within']].assign(panel='b', source='r3_unet/out/C_scores.csv'),
                 sdp.assign(panel='c', source='r3_unet/out/P6_rsa.csv'),
                 pd.DataFrame([dict(panel='ref', dit_ratio_lo=dit_ratio[0], dit_ratio_hi=dit_ratio[1], dit_dR2_lo=dit_dr2[0], dit_dR2_hi=dit_dr2[1],
                                    unrel_rsa_lo=unrel.rsa.min(), unrel_rsa_hi=unrel.rsa.max(), source='lenses/out + r3_unet/out/P6_rsa.csv')])], ignore_index=True)
write_csv(out, 'I-F13_unet')

fig = plt.figure(figsize=(TEXTWIDTH, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.85, 1.05], wspace=0.42)
# (a)
ax = fig.add_subplot(gs[0])
ax.axhspan(*dit_ratio, color=LIGHT, alpha=0.7, lw=0, zorder=0, label='DiT archive medians')
ax.axhline(0.10, color=INK2, lw=0.8, ls='--', zorder=1, label='C1 threshold')
for ch, mk, fc, dx, lb in (('attn', 'o', INK, -0.13, 'attention'), ('mlp', 's', 'white', 0.13, 'MLP')):
    d = agg[agg.channel == ch]
    ax.vlines(d.block + dx, d['min'], d['max'], color=INK if ch == 'attn' else INK2, lw=0.9, zorder=2)
    ax.scatter(d.block + dx, d['median'], marker=mk, s=16, facecolor=fc, edgecolor=INK, lw=0.8, zorder=3, label=lb)
ax.set_yscale('log'); ax.set_ylim(1e-3, 1.0); ax.set_xlim(-0.6, 15.6)
import matplotlib.transforms as mtr
lev = [(0, 1, '320'), (2, 3, '640'), (4, 5, '1280'), (6, 6, 'm'), (7, 9, '1280'), (10, 12, '640'), (13, 15, '320')]
for a, b, t in lev:
    ax.text((a + b) / 2, 1.01, t, fontsize=5.8, color=INK2, ha='center', va='bottom', transform=mtr.blended_transform_factory(ax.transData, ax.transAxes))
for x in (1.5, 3.5, 5.5, 6.5, 9.5, 12.5):
    ax.axvline(x, color='#e2e0db', lw=0.6, zorder=0)
ax.set_xticks(range(0, 16, 3)); ax.set_xlabel('U-Net transformer block (forward order)'); ax.set_ylabel(r'$g\,/\,g_{\rm MP}(N,D)$')
ax.legend(loc='lower left', fontsize=6.0, ncol=2, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
panel_label(ax, 'a')
# (b)
ax = fig.add_subplot(gs[1])
ax.axvspan(*dit_dr2, ymin=0.55, ymax=0.95, color=LIGHT, alpha=0.7, lw=0)
for y, col, thr in ((1, 'C2_dR2', 0.30), (0, 'C3_dR2_within', 0.15)):
    ax.plot([thr, thr], [y - 0.3, y + 0.3], color=INK2, lw=0.8, ls='--')
    ax.scatter(S.loc['attn', col], y + 0.08, marker='o', s=22, color=INK, zorder=3)
    ax.scatter(S.loc['mlp', col], y - 0.08, marker='s', s=20, facecolor='white', edgecolor=INK, lw=0.8, zorder=3)
ax.text(np.mean(dit_dr2), 1.36, 'DiT range', fontsize=6.0, color=INK2, ha='center')
ax.set_yticks([0, 1], ['C3 block\nover level', 'C2 block\nover $g$'], fontsize=6.8); ax.set_ylim(-0.6, 1.6)
ax.set_xlim(0, 1); ax.set_xlabel(r'held-out $\Delta R^2$ for $\beta_{\rm layer}$'); ax.grid(axis='y', visible=False)
ax.text(0.02, -0.5, 'dashed: prediction threshold', fontsize=5.8, color=INK2)
panel_label(ax, 'b')
# (c)
ax = fig.add_subplot(gs[2])
ax.axhspan(unrel.rsa.min(), unrel.rsa.max(), color=LIGHT, alpha=0.7, lw=0)
ax.text(4.45, unrel.rsa.max() + 0.01, 'unrelated DiT pairs', fontsize=6.0, color=INK2, ha='right', va='bottom')
mk = {'first': 'o', 'middle': 'D', 'last': '^'}
for i, m in enumerate(MODELS):
    for j, tau in enumerate(('first', 'middle', 'last')):
        r = sdp[(sdp.a == 'SD15attn') & (sdp.b == m) & (sdp.tau == tau)].rsa.iloc[0]
        ax.scatter(i + (j - 1) * 0.18, r, marker=mk[tau], s=16, color=MODEL_COLOR[m], lw=0, zorder=3, label=tau if i == 0 else None)
ax.set_xticks(range(5), [SHORT[m] for m in MODELS], fontsize=6.6, rotation=30); ax.set_ylim(0, 0.7); ax.set_xlim(-0.6, 4.6)
ax.set_ylabel(r'Mantel $\rho$ with SD1.5 attn')
leg = ax.legend(loc='upper left', fontsize=6.2, ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
for h in leg.legend_handles: h.set_color(INK2)
panel_label(ax, 'c')
save(fig, 'I-F13_unet')
print('I-F13 done')
