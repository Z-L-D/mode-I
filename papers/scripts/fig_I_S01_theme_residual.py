"""I-S1  Prompt/theme residual alignment for all model pairs (register #20: USE WITH WORDING).
Same-theme residual alignment after removing each model/time backbone, first / middle / last step, with the
theme-label permutation null p95 (R6 S2a confirmed for the lineage; S2b REFUTED: unrelated pairs are not near zero).
Source: review/round6/out/S2_theme_residual.csv
"""
from common_I import *

T = pd.read_csv(f'{R6}/S2_theme_residual.csv'); T['source'] = 'round6/out/S2_theme_residual.csv'
write_csv(T, 'I-S1_theme_residual')
order = T.drop_duplicates(['a', 'b'])[['a', 'b', 'kind']].reset_index(drop=True)
TAU = ['first', 'middle', 'last']; TC = ['#9cc0ea', C[0], '#123a6b']; OFF = [-0.22, 0, 0.22]
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for i, r in order.iterrows():
    for k, tau in enumerate(TAU):
        d = T[(T.a == r.a) & (T.b == r.b) & (T.tau == tau)].iloc[0]
        y = i + OFF[k]
        ax.plot([0, d.null_p95], [y, y], color=LIGHT, lw=2.2, solid_capstyle='butt', zorder=1)
        if d.p < 0.05:
            ax.scatter(d.theme_r, y, s=16, color=TC[k], lw=0, zorder=3)
        else:
            ax.scatter(d.theme_r, y, s=16, facecolor='white', edgecolor=TC[k], lw=0.9, zorder=3)
for k, tau in enumerate(TAU):
    ax.scatter([], [], s=16, color=TC[k], label=f'{tau} step')
ax.scatter([], [], s=16, facecolor='white', edgecolor=INK2, label='p ≥ 0.05')
ax.plot([], [], color=LIGHT, lw=2.2, label='null p95')
lab = [pair_label(r.a, r.b) + (' (lineage)' if r.kind == 'lineage' else '') for r in order.itertuples()]
ax.set_yticks(range(len(order)), lab, fontsize=7); ax.invert_yaxis()
ax.set_xlim(0, 0.8); ax.set_xlabel('Same-theme residual alignment r')
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.52), fontsize=6.5, handletextpad=0.3)
save(fig, 'I-S1_theme_residual')
print('I-S1 done; unrelated max', T[T.kind == 'unrelated'].theme_r.max().round(3))
