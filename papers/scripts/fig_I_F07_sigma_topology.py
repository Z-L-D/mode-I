"""I-F7  Sigma-axis three-regime curve in Z-Image MLP; attention contrast (register #1 sigma-axis wording, #2, #3).
(a) Population-mean layer-mean MLP g(sigma) (mean over prompts of the per-prompt mean), scaled to its own peak, with the
    continuous three-segment piecewise-linear fit of R6 S4 (fit3, unchanged): archive ZIT and ZIB, the clean ZIB CFG 4
    cond branch (R4) and ZIB CFG 1 cond-only (R3). '+-+' in 100% of 1000 prompt bootstraps in every channel.
(b) Attention layer-mean g in all five models with the same fit: '++-' (post hoc P1), not '+-+' (S4e refuted).
Sources: recomputed from F_*.pkl, R5_Z40_1024_s1001_*.pkl, F3_Z10_1024_s1001_*.pkl with the S4 fit; checked against
review/round6/out/S4_topology.csv, posthoc_R6C.json (P1), P3_order_perm.csv.
SD1.5 U-Net (post hoc, R3C_ADDENDUM_P7.md, frozen 22:45:30Z): the same S4 analysis on the cond branch, drawn on the
rectified-flow clock at equal signal-to-noise ratio t = sigma_VP / (1 + sigma_VP); MLP in (a), attention in (b).
Sources: review/r3_unet/out/P7_clock_curves.csv, P7_clock_topology.csv (u4_clock_topology.py).
ACE-Step 1.5 turbo (post hoc, R3B_SCORING_PLAN.md): same analysis on its 6 logged flow-matching sigmas; review/r3_ace/out/P7_clock_*.csv.
"""
from common_I import *

rc = cubes()


def fit3(y, s):  # identical to r6c_S4_topology.fit3
    u = -s; T = len(y); best = None
    for i in range(1, T - 2):
        for j in range(i + 1, T - 1):
            A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None); sse = ((A @ coef - y) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, i, j, coef)
    sse, i, j, c = best; sl = np.array([c[1], c[1] + c[2], c[1] + c[2] + c[3]])
    A = np.stack([np.ones(T), u, np.maximum(0, u - u[i]), np.maximum(0, u - u[j])], 1)
    return ''.join('+' if v > 0 else '-' for v in sl), i, j, A @ c


def per_prompt_mean(X, meta):  # as r6c_S4_topology.per_prompt, then mean over prompts
    k = meta.pt.astype(str).str.strip().values; up = np.unique(k)
    return np.stack([X[k == p][..., 0].mean((0, 2)) for p in up]).mean(0)


S4 = pd.read_csv(f'{R6}/S4_topology.csv').set_index('set')
P1 = rjson(f'{R6}/posthoc_R6C.json')['P1_attention_topology']
P3 = pd.read_csv(f'{R6}/P3_order_perm.csv').set_index('set')
z4 = rc.R5('Z40_1024_s1001', 'mlp'); _, m3, X3, s3 = rc.R3('Z10_1024_s1001', 'mlp')
MLP = [('archive ZIT-mlp', 'ZIT archive (8 steps)', *rc.arch('ZIT', 'mlp')[1:]),
       ('archive ZIB-mlp', 'ZIB archive, pooled CFG 4', *rc.arch('ZIB', 'mlp')[1:]),
       ('R4 ZIB CFG4 cond-mlp', 'ZIB CFG 4 cond branch', z4['meta'], z4['X']['cond'], np.asarray(z4['sigma'], float)),
       ('R3 ZIB CFG1 cond-only-mlp', 'ZIB CFG 1 cond-only', m3, X3, np.asarray(s3, float))]
rows = []
for key, lab, meta, X, s in MLP:
    y = per_prompt_mean(X, meta); top, i, j, fit = fit3(y, s)
    ref = S4.loc[key]; assert top == ref.topology and abs(s[i] - ref.break1_sigma) < 1e-6 and abs(s[j] - ref.break2_sigma) < 1e-6, key
    pk = y.max()
    for t in range(len(s)):
        rows.append(dict(panel='a', set=key, label=lab, sigma=s[t], g=y[t], g_scaled=y[t] / pk, fit_scaled=fit[t] / pk, topology=top,
                         break1_sigma=s[i], break2_sigma=s[j], frac_pmp_boot=ref.frac_pmp, contraction=ref.contraction,
                         order_perm_n_ge=P3.loc[key, 'n_null_ge_obs'] if key in P3.index else np.nan, source='recomputed; checked vs round6/out/S4_topology.csv'))
for m in MODELS:
    key = f'archive {m}-attn'; g, meta, X, s = rc.arch(m, 'attn'); y = per_prompt_mean(X, meta); top, i, j, fit = fit3(y, s)
    ref = S4.loc[key]; assert top == ref.topology == '++-'
    for t in range(len(s)):
        rows.append(dict(panel='b', set=key, label=SHORT[m], sigma=s[t], g=y[t], g_scaled=y[t] / y.max(), fit_scaled=fit[t] / y.max(), topology=top,
                         break1_sigma=s[i], break2_sigma=s[j], modal_frac_boot=P1[m]['modal_frac'], late_drop=P1[m]['late_drop'],
                         source='recomputed; checked vs round6/out/S4_topology.csv, posthoc_R6C.json P1'))
U7 = pd.read_csv(f'{REVIEW}/r3_unet/out/P7_clock_curves.csv'); T7 = pd.read_csv(f'{REVIEW}/r3_unet/out/P7_clock_topology.csv')
T7 = T7[T7.clock == 't=sigma/(1+sigma)'].set_index('channel')
for sub, pan in (('mlp', 'a'), ('attn', 'b')):
    d = U7[U7.channel == f'SD1.5-{sub}']; r = T7.loc[f'SD1.5-{sub}']; pk = d.g.max()
    for q in d.itertuples():
        rows.append(dict(panel=pan, set=f'SD1.5-{sub}', label=f'SD1.5 {sub}', sigma=q.t, g=q.g, g_scaled=q.g / pk, fit_scaled=q.fit / pk, topology=r.topology,
                         break1_sigma=r.break1, break2_sigma=r.break2, modal_frac_boot=r.modal_frac, frac_pmp_boot=r.frac_pmp, contraction=r.contraction,
                         source='review/r3_unet/out/P7_clock_curves.csv (post hoc; x = t = sigma_VP/(1+sigma_VP))'))
V7 = pd.read_csv(f'{REVIEW}/r3_ace/out/P7_clock_curves.csv'); W7 = pd.read_csv(f'{REVIEW}/r3_ace/out/P7_clock_topology.csv').set_index('channel')
for sub, pan in (('mlp', 'a'), ('attn', 'b')):
    d = V7[V7.channel == f'ACE-{sub}']; r = W7.loc[f'ACE-{sub}']; pk = d.g.max()
    for q in d.itertuples():
        rows.append(dict(panel=pan, set=f'ACE-{sub}', label=f'ACE-Step {sub}', sigma=q.sigma, g=q.g, g_scaled=q.g / pk, fit_scaled=q.fit / pk, topology=r.topology,
                         break1_sigma=r.break1, break2_sigma=r.break2, modal_frac_boot=r.modal_frac, frac_pmp_boot=r.frac_pmp, contraction=r.contraction,
                         source='review/r3_ace/out/P7_clock_curves.csv (post hoc; 6 steps, flow-matching sigma)'))
D = pd.DataFrame(rows)
write_csv(D, 'I-F7_sigma_three_regime')

fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.75), gridspec_kw=dict(wspace=0.28))
ax = axs[0]
MC = [C[0], C[1], INK, GRAY]; MK = ['o', 's', 'D', '^']
for (key, lab, *_), col, mk in zip(MLP, MC, MK):
    d = D[(D.panel == 'a') & (D.set == key)]
    ax.plot(d.sigma, d.fit_scaled, color=col, lw=1.0, zorder=2)
    ax.scatter(d.sigma, d.g_scaled, s=9, marker=mk, color=col, lw=0, zorder=3, label=f'{lab}')
d = D[(D.panel == 'a') & (D.set == 'SD1.5-mlp')]
ax.plot(d.sigma, d.fit_scaled, color=INK, lw=0.9, ls='--', zorder=2)
ax.scatter(d.sigma, d.g_scaled, s=9, marker='v', facecolor='white', edgecolor=INK, lw=0.6, zorder=3,
           label=f"SD1.5 U-Net, t clock ({d.topology.iloc[0]})")
d = D[(D.panel == 'a') & (D.set == 'ACE-mlp')]
ax.plot(d.sigma, d.fit_scaled, color=INK2, lw=0.9, ls='-.', zorder=2)
ax.scatter(d.sigma, d.g_scaled, s=14, marker='D', facecolor='white', edgecolor=INK2, lw=0.7, zorder=3, label=f"ACE-Step, 6 steps ({d.topology.iloc[0]})")
ax.set_xlim(1.03, -0.03); ax.set_xlabel('σ   (SD1.5: t = σ/(1+σ))'); ax.set_ylabel('MLP layer-mean g / own peak')
ax.set_ylim(0.0, 1.08)
ax.legend(loc='lower right', fontsize=6.0, handletextpad=0.2, markerscale=1.3, borderaxespad=0.1)
panel_label(ax, 'a')

ax = axs[1]
for m in MODELS:
    d = D[(D.panel == 'b') & (D.set == f'archive {m}-attn')]
    ax.plot(d.sigma, d.fit_scaled, color=MODEL_COLOR[m], lw=1.0, zorder=2)
    ax.scatter(d.sigma, d.g_scaled, s=6, color=MODEL_COLOR[m], lw=0, zorder=3, label=f"{SHORT[m]}  (++− {100 * P1[m]['modal_frac']:.0f}%)")
d = D[(D.panel == 'b') & (D.set == 'SD1.5-attn')]
ax.plot(d.sigma, d.fit_scaled, color=INK, lw=1.0, ls='--', zorder=2)
ax.scatter(d.sigma, d.g_scaled, s=6, color=INK, lw=0, zorder=3, label=f"SD1.5, t clock ({d.topology.iloc[0]} {100 * d.modal_frac_boot.iloc[0]:.0f}%)")
d = D[(D.panel == 'b') & (D.set == 'ACE-attn')]
ax.plot(d.sigma, d.fit_scaled, color=INK2, lw=1.0, ls='-.', zorder=2)
ax.scatter(d.sigma, d.g_scaled, s=12, marker='D', color=INK2, lw=0, zorder=3, label=f"ACE-Step ({d.topology.iloc[0]} {100 * d.modal_frac_boot.iloc[0]:.0f}%)")
ax.set_xlim(1.03, -0.03); ax.set_xlabel('σ   (SD1.5: t = σ/(1+σ))'); ax.set_ylabel('attention layer-mean g / own max')
ax.legend(loc='lower right', fontsize=6.4, handletextpad=0.2, markerscale=1.6, title="modal fit in bootstraps", title_fontsize=6.4)
panel_label(ax, 'b')
save(fig, 'I-F7_sigma_three_regime')
print('I-F7 done')
