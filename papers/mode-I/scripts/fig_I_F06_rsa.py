"""I-F6  Cross-model prompt geometry (register #19, preregistered; #20 for theme wording).
(a-c) Mantel rho between the 117x117 prompt-dissimilarity matrices of every model pair at the first, middle and last
      sampling step; every cell p = 0.0005 against its own 2000-permutation null (R6 S2c).
(d)   Robustness (R6 P2, post hoc): within-theme RSA (theme means removed, theme-preserving null) and partial Mantel
      controlling prompt-text RSA, against the full RSA, all 10 pairs x 3 steps.
Sources: review/round6/out/S2_rsa.csv, P2_rsa_robustness.csv
Bottom row of (a-c), separated by a line: SD1.5 U-Net attention (cond) against each DiT, post hoc, on the 116 prompts shared
with every archive model (review/r3_unet/out/P6_rsa.csv, same distance and Mantel test); the DiT cells stay on 117 prompts.
"""
from matplotlib.colors import LinearSegmentedColormap
from common_I import *

RS = pd.read_csv(f'{R6}/S2_rsa.csv'); P2 = pd.read_csv(f'{R6}/P2_rsa_robustness.csv')
M = RS.merge(P2, on=['a', 'b', 'kind', 'tau'])
M['source'] = 'round6/out/S2_rsa.csv + P2_rsa_robustness.csv'
U6 = pd.read_csv(f'{REVIEW}/r3_unet/out/P6_rsa.csv'); U6 = U6[(U6.a == 'SD15attn') & U6.b.isin(MODELS)]
assert (U6.p <= 0.0005 + 1e-9).all() and len(U6) == 15
write_csv(pd.concat([M, U6.assign(kind='U-Net vs DiT (post hoc, 116 prompts)', source='review/r3_unet/out/P6_rsa.csv')], ignore_index=True), 'I-F6_prompt_rsa')
assert (RS.p <= 0.0005 + 1e-9).all()

cmap = LinearSegmentedColormap.from_list('seq', ['#f4f7fb', '#9cc0ea', C[0], '#123a6b'])
fig = plt.figure(figsize=(TEXTWIDTH, 3.4))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 0.05, 0.62, 1.0], wspace=0.1)
names = [SHORT[m] for m in MODELS] + ['SD1.5 attn']
for k, tau in enumerate(['first', 'middle', 'last']):
    ax = fig.add_subplot(gs[k])
    A = np.full((6, 6), np.nan)
    for r in RS[RS.tau == tau].itertuples():
        i, j = MODELS.index(r.a), MODELS.index(r.b); A[max(i, j), min(i, j)] = r.rsa
    for r in U6[U6.tau == tau].itertuples():
        A[5, MODELS.index(r.b)] = r.rsa
    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=0.65)
    for i in range(6):
        for j in range(i):
            v = A[i, j]
            ax.text(j, i, f'{v:.2f}'.replace('0.', '.', 1), ha='center', va='center', fontsize=6.6, color='white' if v > 0.4 else INK,
                    fontweight='bold' if (i, j) == (1, 0) else 'normal')
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=INK, lw=1.0))   # ZIB-ZIT (lineage) cell
    ax.plot([-0.5, 4.5], [4.5, 4.5], color=INK, lw=0.8)   # U-Net row (post hoc, 116 prompts) below the line
    ax.set_xticks(range(5), names[:5], rotation=50, ha='right', fontsize=6.5)
    ax.set_yticks(range(1, 6), names[1:] if k == 0 else [''] * 5, fontsize=6.5)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(5.5, 0.5)
    ax.grid(False); [sp.set_visible(False) for sp in ax.spines.values()]
    ax.tick_params(length=0)
    ax.text(0.98, 0.98, f'{tau} step', transform=ax.transAxes, ha='right', va='top', fontsize=7.0, color=INK)
    panel_label(ax, 'abc'[k])
cax = fig.add_subplot(gs[3]); cb = fig.colorbar(im, cax=cax); cax.set_title('ρ', fontsize=7, color=INK); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)

ax = fig.add_subplot(gs[5])
lin = M.kind == 'lineage'
for mk, col_k, lb in [('within_theme_rsa', 'o', 'within theme'), ('partial_text_rsa', 'D', 'partial, text RSA')]:
    sig = M[{'within_theme_rsa': 'within_p', 'partial_text_rsa': 'partial_p'}[mk]] < 0.05
    for sel, col in [(~lin, GRAY), (lin, C[0])]:
        d = M[sel]; s_ = sig[sel]
        ax.scatter(d.rsa[s_], d[mk][s_], marker=col_k, s=13 if col_k == 'o' else 11, color=col, lw=0, zorder=3)
        ax.scatter(d.rsa[~s_], d[mk][~s_], marker=col_k, s=13, facecolor='white', edgecolor=col, lw=0.8, zorder=3)
ax.plot([0, 0.7], [0, 0.7], color=INK2, lw=0.7, ls='--', zorder=1)
ax.scatter([], [], marker='o', s=13, color=INK2, label='within theme'); ax.scatter([], [], marker='D', s=11, color=INK2, label='partial, text RSA')
ax.scatter([], [], marker='o', s=13, facecolor='white', edgecolor=INK2, label='p ≥ 0.05')
ax.text(0.62, 0.50, 'ZIT–ZIT-25', color=INK, fontsize=6.8, ha='center')
ax.set_xlim(0, 0.68); ax.set_ylim(0, 0.80)
ax.set_xlabel('full Mantel ρ'); ax.set_ylabel('controlled ρ', labelpad=1)
ax.legend(loc='upper left', fontsize=6.3, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax, 'd')
save(fig, 'I-F6_prompt_rsa')
print('I-F6 done; unrelated range', RS[RS.kind == 'unrelated'].rsa.min().round(3), RS[RS.kind == 'unrelated'].rsa.max().round(3))
