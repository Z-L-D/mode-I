"""I-S2  Step-count and sampler controls at matched sigma (register #4: USE WITH WORDING,
"Per-sigma geometry changes <= 2-3% between 8/40 and 100 steps"; disclose the 4-generation ZIT control).
(a) SD3.5 M one-prompt pyramid, explicit cond branch: layer-mean attention g(sigma) at 40 and 100 steps, CFG 1 and CFG 4.5
    (recomputed from the R4 cubes). R5 Q5: same-prompt 40->100 shift 0.0326 at CFG 1 (seed-split null p95 0.0266,
    p = 0.023) and 0.0068 at CFG 4.5 cond (p95 0.0216).
(b) Mochi 30 vs 100 steps, per-sigma path speed by sigma band (R2 E4b). The arms also differ in CFG (4.5 vs 1.0).
(c) ZIT 8 vs 100 steps at matched sigma: median relative difference of g, gap and H (R2 E4; the 100-step set is 4 generations).
Sources: cubes R5_P10_40/100_attn.pkl, R5_P45_40/100_attn.pkl; review/round5/out/Q_results.json; review/round2/out/E4b_mochi_speed.csv, E4_stepcount.csv
"""
from common_I import *

rc = cubes()
Q = rjson(f'{R5}/Q_results.json')
rows = []
for arm, cfg in [('P10', 'CFG 1'), ('P45', 'CFG 4.5 cond')]:
    for n in (40, 100):
        d = rc.R5(f'{arm}_{n}', 'attn'); E = d['X']['cond'][..., 0].mean((0, 2)); s = np.asarray(d['sigma'], float)
        for t in range(len(s)):
            rows.append(dict(panel='a', arm=cfg, steps=n, gens=len(d['X']['cond']), sigma=s[t], g_layer_mean=E[t], source=f'recomputed from R5_{arm}_{n}_attn.pkl (cond)'))
for k in ['CFG1', 'CFG45_cond']:
    rows.append(dict(panel='a_stat', arm=k, observed=Q[f'Q5_{k}_observed'], null_p95=Q[f'Q5_{k}_null_p95'], perm_p=Q[f'Q5_{k}_perm_p'], source='round5/out/Q_results.json Q5'))
E4b = pd.read_csv(f'{R2}/E4b_mochi_speed.csv')
for r in E4b.itertuples():
    rows.append(dict(panel='b', band=r.band, speed_30=r.speed_30, speed_100=r.speed_100, ratio=r.ratio, source='round2/out/E4b_mochi_speed.csv'))
E4 = pd.read_csv(f'{R2}/E4_stepcount.csv'); E4 = E4[E4.ref == 'matched']
for r in E4.itertuples():
    rows.append(dict(panel='c', sub=r.sub, var=r.var, median_rel_diff=r.median, p90=r.p90, source='round2/out/E4_stepcount.csv (ref = matched)'))
D = pd.DataFrame(rows)
write_csv(D, 'I-S2_stepcount_controls')

fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1, 0.9], wspace=0.5))
ax = axs[0]
for arm, col in [('CFG 1', MODEL_COLOR['SD35M']), ('CFG 4.5 cond', INK)]:
    d100 = D[(D.panel == 'a') & (D.arm == arm) & (D.steps == 100)]; d40 = D[(D.panel == 'a') & (D.arm == arm) & (D.steps == 40)]
    ax.plot(d100.sigma, d100.g_layer_mean, color=col, lw=1.1)
    ax.scatter(d40.sigma, d40.g_layer_mean, s=8, facecolor='white', edgecolor=col, lw=0.7, zorder=3)
    xt, yt = {'CFG 1': (0.42, 0.0565), 'CFG 4.5 cond': (0.62, 0.0668)}[arm]
    ax.text(xt, yt, arm, fontsize=6.6, color=INK, ha='center', va='center')
ax.plot([], [], color=INK2, lw=1.1, label='100 steps'); ax.scatter([], [], s=8, facecolor='white', edgecolor=INK2, lw=0.7, label='40 steps')
ax.invert_xaxis(); ax.set_xlabel('σ'); ax.set_ylabel('SD3.5 M layer-mean g')
ax.legend(loc='lower right', fontsize=6.4, handletextpad=0.3)
panel_label(ax, 'a')

ax = axs[1]
b = D[D.panel == 'b'].copy(); b['lo'] = b.band.str.split('-').str[0].astype(float); b = b.sort_values('lo', ascending=False)
yb = np.arange(len(b))
ax.scatter(b.speed_30, yb, s=16, color=MODEL_COLOR['MOCHI'], lw=0, zorder=3, label='30 steps, CFG 4.5')
ax.scatter(b.speed_100, yb, s=16, facecolor='white', edgecolor=MODEL_COLOR['MOCHI'], lw=0.9, zorder=3, label='100 steps, CFG 1')
ax.set_xscale('log'); ax.set_yticks(yb, [x.replace('-1.01', '–1.0').replace('-', '–') for x in b.band], fontsize=6.6); ax.set_ylim(len(b) - 0.5, -0.6)
ax.set_xlabel('Mochi path speed |dA/dσ|')
ax.legend(loc='lower right', fontsize=6.2, handletextpad=0.2)
panel_label(ax, 'b')

ax = axs[2]
c = D[D.panel == 'c']; VAR = [('g_micro', 'g'), ('spectral_gap', 'gap'), ('sv_entropy', 'H')]
for k, (v, lb) in enumerate(VAR):
    for sub, off, col in [('attn', -0.13, C[0]), ('mlp', 0.13, INK)]:
        val = c[(c['var'] == v) & (c['sub'] == sub)].median_rel_diff.iloc[0] * 100
        ax.barh(k + off, val, height=0.24, color=col, lw=0)
        ax.text(val + 0.05, k + off, f'{val:.2f}%', va='center', fontsize=6.2, color=INK)
ax.set_yticks(range(3), [v[1] for v in VAR]); ax.set_ylim(2.6, -0.6); ax.set_xlim(0, 2.8)
ax.set_xlabel('ZIT 8 vs 100: median |Δ| (%)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C[0], label='attn'), Patch(color=INK, label='MLP')], loc='lower right', fontsize=6.4, handletextpad=0.3)
ax.grid(axis='y', visible=False)
panel_label(ax, 'c')
save(fig, 'I-S2_stepcount_controls')
print('I-S2 done')
