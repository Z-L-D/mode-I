"""I-F8  Layer-axis beta_layer topology across architectures (register #1 layer-axis wording; post hoc analyzer replay A2-A4).
(a) Crossing pattern of the analyzer's beta_layer(g) = g(L+1) - g(L) at each denoising step (one sigma at a time).
(b) Fraction of 100 generation bootstraps with the + - + - three-crossing pattern on all steps pooled, and how often the
    09-15 Surrogate A (per-layer marginals) and Surrogate B (mean field g(L, sigma)) reproduce the observed pattern (30 draws).
(c) The binned beta_layer(g) curve itself (all steps pooled), as the analyzer draws it: 29 bins in g, >= 10 points per bin,
    mean +/- 1 s.d.; dashed lines at the analyzer's crossings g*.
Sources: review/round6/out/A4_beta_layer_per_step.csv, A2_beta_layer_bootstrap.csv, A4_beta_layer_surrogates.csv;
         data/I-F8c_beta_curves.csv and I-F8c_beta_crossings.csv (written by extract_I_F08_beta_curves.py with the analyzer).
"""
from common_I import *

PS = pd.read_csv(f'{R6}/A4_beta_layer_per_step.csv'); SU = pd.read_csv(f'{R6}/A4_beta_layer_surrogates.csv').set_index('channel')
BO = pd.read_csv(f'{R6}/A2_beta_layer_bootstrap.csv')
CH = [('SD35M archive attn', 'SD3.5 M attn', ('SD35M archive', 'as loaded (pooled)')),
      ('SD35L archive attn', 'SD3.5 L attn', ('SD35L archive', 'as loaded (pooled)')),
      ('ZIB archive (pooled CFG4) attn', 'ZIT-25 attn', ('ZIB archive (pooled CFG4)', 'attn')),
      ('ZIB archive (pooled CFG4) mlp', 'ZIT-25 MLP', ('ZIB archive (pooled CFG4)', 'mlp')),
      ('ZIT archive mlp', 'ZIT MLP', ('ZIT archive', 'mlp')),
      ('MOCHI archive CFG4.5 attn', 'Mochi attn', ('MOCHI archive CFG4.5', 'as loaded (pooled)')),
      ('ZIT archive attn', 'ZIT attn', ('ZIT archive', 'attn'))]
rows = []
for ch, lab, (bset, bsub) in CH:
    for r in PS[PS.channel == ch].itertuples():
        rows.append(dict(panel='a', channel=ch, label=lab, step=int(r.step), sigma=r.sigma, pattern=r.pattern if isinstance(r.pattern, str) else '',
                         n_cross=int(r.n), three_cross_pmpm=int(r.pattern == '+- -+ +-'), source='round6/out/A4_beta_layer_per_step.csv'))
    b = BO[(BO.set == bset) & (BO.submodule == bsub)].iloc[0]; s = SU.loc[ch]
    rows.append(dict(panel='b', channel=ch, label=lab, gens=b.gens, observed_pattern=b.observed_pattern, boot_frac_pmpm=b.frac_3_plus_minus_plus,
                     boot_frac_1=b.frac_1, boot_frac_5plus=b.frac_5plus, per_step_frac_3cross=s.per_step_frac_3cross,
                     surrA_reproduces_obs=s.surrA_frac_obs, surrB_reproduces_obs=s.surrB_frac_obs, surrB_frac_3=s.surrB_frac_3,
                     source='round6/out/A2_beta_layer_bootstrap.csv + A4_beta_layer_surrogates.csv'))
D = pd.DataFrame(rows)
write_csv(D, 'I-F8_layer_beta_topology')

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(TEXTWIDTH, 4.75))
GS = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.7, 1.35], hspace=0.75)
top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=GS[0], width_ratios=[1.4, 1], wspace=0.06)
axs = [fig.add_subplot(top[0]), fig.add_subplot(top[1])]
ax = axs[0]
for y, (ch, lab, _) in enumerate(CH):
    d = D[(D.panel == 'a') & (D.channel == ch)]
    for r in d.itertuples():
        if r.three_cross_pmpm:
            ax.scatter(r.sigma, y, s=17, color=C[0], lw=0, zorder=3)
        elif r.n_cross == 1:
            ax.scatter(r.sigma, y, s=13, color=GRAY, lw=0, zorder=3)
        else:
            ax.scatter(r.sigma, y, s=13, facecolor='white', edgecolor=INK2, lw=0.7, zorder=3)
ax.set_yticks(range(len(CH)), [c[1] for c in CH], fontsize=7.2); ax.set_ylim(len(CH) - 0.5, -0.6)
ax.set_xlim(1.03, 0.02); ax.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2]); ax.set_xlabel('σ (one step at a time)')
ax.grid(axis='y', visible=False)
ax.scatter([], [], s=17, color=C[0], label='+ − + − (3 crossings)'); ax.scatter([], [], s=13, color=GRAY, label='1 crossing')
ax.scatter([], [], s=13, facecolor='white', edgecolor=INK2, label='other count (0, 2, 4, 5, 7)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=6.6, handletextpad=0.2, columnspacing=0.9)
panel_label(ax, 'a')

ax = axs[1]
b = D[D.panel == 'b'].reset_index(drop=True)
ax.barh(np.arange(len(b)) + 0.14, b.boot_frac_pmpm * 100, height=0.3, color=C[0], edgecolor='white', linewidth=0)
for j, r in b.iterrows():
    v = r.boot_frac_pmpm * 100
    if r.observed_pattern == '+- -+ +-':
        A_ = int(round(r.surrA_reproduces_obs * 30)); B_ = int(round(r.surrB_reproduces_obs * 30))
        verdict = 'null-generic' if (A_ == 30 and B_ == 30) else ('beats A and B' if (A_ == 0 and B_ == 0) else 'beats A, not B')
        txt = f'{v:.0f}%   surr. A {A_}/30, B {B_}/30: {verdict}'
    else:
        txt = f'{v:.0f}%   all steps pooled: + − only'
    ax.text(0.5, j - 0.2, txt, va='center', ha='left', fontsize=6.3, color=INK)
ax.set_yticks(range(len(b)), [''] * len(b)); ax.set_ylim(len(b) - 0.5, -0.6)
ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel('bootstraps with + − + − (%)')
ax.grid(axis='y', visible=False)
panel_label(ax, 'b')

# (c) the curves themselves, read from the analyzer's own plot_beta_layer
CV = pd.read_csv(os.path.join(DATA, 'I-F8c_beta_curves.csv')); CR = pd.read_csv(os.path.join(DATA, 'I-F8c_beta_crossings.csv'))
bot = gridspec.GridSpecFromSubplotSpec(1, len(CH), subplot_spec=GS[1], wspace=0.55)
for k, (ch, lab, _) in enumerate(CH):
    ax = fig.add_subplot(bot[k]); d = CV[CV.channel == lab]
    if d.empty: continue
    x = d.g_center.values * 100; m = d.beta_mean.values * 1e3
    ax.fill_between(x, d.beta_lo_1sd.values * 1e3, d.beta_hi_1sd.values * 1e3, color=LIGHT, lw=0, zorder=1)
    ax.axhline(0, color=INK2, lw=0.6, zorder=2)
    for r in CR[CR.channel == lab].itertuples():
        ax.axvline(r.g_star * 100, color=(C[1] if r.direction == 'pos→neg' else C[2]), lw=0.8, ls='--', zorder=2)
    ax.plot(x, m, color=C[0], lw=1.4, zorder=3)
    span = np.nanmax(np.abs(m)); ax.set_ylim(-1.6 * span, 1.6 * span)
    ns = len(CR[CR.channel == lab]); ax.set_title(f'{lab}\n{ns} crossing' + ('s' if ns != 1 else ''), fontsize=6.3, pad=2, linespacing=1.1); ax.tick_params(labelsize=5.6, pad=1)
    ax.locator_params(axis='x', nbins=3); ax.locator_params(axis='y', nbins=3); ax.grid(False)
    if k == 0:
        ax.set_ylabel('β$_{layer}$ (×10$^{-3}$)', fontsize=6.6); ax.text(-0.75, 1.32, 'c', transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)
    if k == len(CH) // 2:
        ax.set_xlabel('g (×10$^{-2}$)', fontsize=6.8)
save(fig, 'I-F8_layer_beta_topology')
print('I-F8 done')
