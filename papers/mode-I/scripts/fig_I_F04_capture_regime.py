"""I-F4  Scalar projection and capture-regime reversal (register #21; #22 = worked example only, NOT architecture evidence).
(a) Layer-mean attention g(sigma), each model scaled to its own maximum: every envelope rises (recomputed from the archive
    cubes; these are the D1 envelopes of R6 S1).
(b,c) Envelope correlation ZIB<->ZIT vs ZIB<->Mochi when only the ZIB capture is substituted (R6 S1d), on linear g (b)
    and log g (c). The historical 0.9405 > 0.9325 ordering appears only for the pooled archive on linear g.
Sources: review/round6/out/S1_results.json (S1d), S1_historical_reproduction.csv; raw cubes F_*_attn.pkl (panel a)
"""
from common_I import *

rc = cubes()
rows = []
for m in MODELS:
    g, meta, X, s = rc.arch(m, 'attn'); E = X[..., 0].mean((0, 2))
    for t in range(len(s)):
        rows.append(dict(panel='a', model=m, sigma=s[t], g_layer_mean=E[t], g_scaled=E[t] / E.max(), source='recomputed from F_{m}_attn.pkl'.format(m=m)))
S1 = rjson(f'{R6}/S1_results.json')['S1d']
ORDER = [('archive ZIB (pooled, CFG4, 117 prompts)', 'Archive pooled (117)'),
         ('archive ZIB restricted to the R4 CFG4 prompts (pooled)', 'Archive pooled (same 19)'),
         ('R4 ZIB CFG4 uncond (explicit, 19 prompts)', 'CFG 4 uncond (19)'),
         ('R4 ZIB CFG4 branch-mean (synthetic, 19 prompts)', 'CFG 4 branch mean (19)'),
         ('R4 ZIB CFG4 cond (explicit, 19 prompts)', 'CFG 4 cond (19)'),
         ('R4 ZIB CFG1 cond (20 prompts)', 'CFG 1 cond (20)'),
         ('R3 ZIB CFG1 cond-only (60 prompts)', 'CFG 1 cond-only (60)')]
for key, lab in ORDER:
    for obs in ['D1', 'D2']:
        v = S1[f'{key} | {obs}']
        rows.append(dict(panel='b' if obs == 'D1' else 'c', capture=key, label=lab, observable='linear g' if obs == 'D1' else 'log g',
                         r_ZIB_ZIT=v['r_ZIT'], r_ZIB_MOCHI=v['r_MOCHI'], delta=v['delta'], source='round6/out/S1_results.json S1d'))
D = pd.DataFrame(rows)
write_csv(D, 'I-F4_capture_regime_reversal')
hist = pd.read_csv(f'{R6}/S1_historical_reproduction.csv')

fig = plt.figure(figsize=(TEXTWIDTH, 2.7))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.62, 0.8, 0.8], wspace=0.1)
ax = fig.add_subplot(gs[0])
for m in MODELS:
    d = D[(D.panel == 'a') & (D.model == m)]
    ax.plot(d.sigma, d.g_scaled, color=MODEL_COLOR[m], lw=1.3, label=SHORT[m])
ax.set_xticks([1, 0.75, 0.5, 0.25]); ax.invert_xaxis(); ax.set_xlabel('σ'); ax.set_ylabel('attention layer-mean g / own max')
ax.legend(loc='lower right', fontsize=6.6, handlelength=1.4)
panel_label(ax, 'a')
pos_a = ax.get_position()

y = np.arange(len(ORDER))
for k, (pn, obs) in enumerate([('b', 'linear g'), ('c', 'log g')]):
    ax = fig.add_subplot(gs[2 + k])
    d = D[D.panel == pn].reset_index(drop=True)
    for j, r in d.iterrows():
        ax.plot([r.r_ZIB_MOCHI, r.r_ZIB_ZIT], [j, j], color=LIGHT, lw=1.2, zorder=1)
    ax.scatter(d.r_ZIB_ZIT, y, s=22, color=MODEL_COLOR['ZIT'], lw=0, zorder=3, label='ZIT-25–ZIT')
    ax.scatter(d.r_ZIB_MOCHI, y, s=22, marker='s', color=MODEL_COLOR['MOCHI'], lw=0, zorder=3, label='ZIT-25–Mochi')
    ax.set_xlim(0.7, 1.01); ax.set_ylim(len(ORDER) - 0.5, -0.9)
    ax.set_yticks(y, [o[1] for o in ORDER] if k == 0 else [''] * len(y), fontsize=6.8)
    ax.text(0.71, -0.75, f'envelope r, {obs}', fontsize=7.2, color=INK, va='center')
    ax.set_xlabel('Pearson r')
    if k == 0:
        ax.annotate('0.9405 > 0.9325\n(historical)', xy=(0.937, 0), xytext=(0.74, 1.05), fontsize=6.3, color=INK2,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6))
    if k == 1:
        ax.legend(loc='upper center', bbox_to_anchor=(-0.05, -0.2), ncol=2, fontsize=6.8, handletextpad=0.2)
    panel_label(ax, pn)
save(fig, 'I-F4_capture_regime_reversal')
print('I-F4 done'); print(hist.round(4).to_string())
