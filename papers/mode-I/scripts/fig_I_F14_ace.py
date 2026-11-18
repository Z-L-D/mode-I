"""I-F14  Out-of-modality check: ACE-Step 1.5 turbo (audio flow transformer, 6-step distilled), R3 Part B (2026-10-09).
(a) B2 / B5 (preregistered): per-layer g / g_MP(2250, 2048), median over 120 gens (markers) and range over the 6 steps (bars),
    even layers filled, odd layers open; transformer-archive median ratios as a band; the 0.10 prediction threshold.
(b) B6 (preregistered) and replay (post hoc): the analyzer's binned beta_layer(g) curves (plot_beta_layer, unchanged) for ACE
    attention, ACE MLP and the attention even-layer subsequence (descriptive).
(c) B9 (preregistered): within-prompt variance of layer-mean logit g along sigma, with the B9 verdict.
Sources: review/r3_ace/out/cache_{attn,mlp}.npz, B_details.json, beta_curves.csv, A2_bootstrap.csv; review/lenses/out/PH_isotropy_baseline.csv
"""
import json
from common_I import *

U = f'{REVIEW}/r3_ace/out'; LZ = f'{REVIEW}/lenses/out'
DET = json.load(open(f'{U}/B_details.json')); gmp = DET['g_MP']
iso = pd.read_csv(f'{LZ}/PH_isotropy_baseline.csv'); dit_ratio = (iso.ratio_median_to_iso.min(), iso.ratio_median_to_iso.max())
CV = pd.read_csv(f'{U}/beta_curves.csv'); A2 = pd.read_csv(f'{U}/A2_bootstrap.csv').set_index('channel')
rows = []
for s in ('attn', 'mlp'):
    c = np.load(f'{U}/cache_{s}.npz', allow_pickle=True); r = np.median(c['g_gen'], 0) / gmp     # [step, layer]
    for l in range(r.shape[1]):
        rows.append(dict(panel='a', channel=s, layer=l, parity='even' if l % 2 == 0 else 'odd', median=float(np.median(r[:, l])), min=float(r[:, l].min()), max=float(r[:, l].max()),
                         source='r3_ace/out/cache_*.npz; g_MP from B_details.json'))
    for t, (sg, v) in enumerate(zip(c['sigma'], DET['B9'][s]['curve'])):
        rows.append(dict(panel='c', channel=s, step=t, sigma=float(sg), chi_logit=v, source='r3_ace/out/B_details.json B9'))
A = pd.DataFrame(rows)
write_csv(pd.concat([A, CV.assign(panel='b', source='r3_ace/out/beta_curves.csv')], ignore_index=True), 'I-F14_ace')

fig = plt.figure(figsize=(TEXTWIDTH, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.05, 0.95], wspace=0.55)
ax = fig.add_subplot(gs[0])
ax.axhspan(*dit_ratio, color=LIGHT, alpha=0.7, lw=0, zorder=0, label='image-transformer medians')
ax.axhline(0.10, color=INK2, lw=0.8, ls='--', zorder=1, label='B2 threshold')
for s, mk, dx in (('attn', 'o', -0.15), ('mlp', 's', 0.15)):
    d = A[(A.panel == 'a') & (A.channel == s)]
    ax.vlines(d.layer + dx, d['min'], d['max'], color=INK2, lw=0.8, zorder=2)
    for par, fc in (('even', INK2), ('odd', 'white')):
        e = d[d.parity == par]
        ax.scatter(e.layer + dx, e['median'], marker=mk, s=15, facecolor=fc, edgecolor=INK2, lw=0.8, zorder=3,
                   label=f"{'attention' if s == 'attn' else 'MLP'}, {par} layers")
ax.set_yscale('log'); ax.set_ylim(1e-3, 1.0); ax.set_xlim(-0.8, 23.8); ax.set_xticks(range(0, 24, 4))
ax.set_xlabel('ACE-Step layer'); ax.set_ylabel(r'$g\,/\,g_{\rm MP}(N,D)$')
fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.3, -0.07), fontsize=6.2, ncol=3, handletextpad=0.3, columnspacing=1.0)
panel_label(ax, 'a')
ax = fig.add_subplot(gs[1])
ax.axhline(0, color=INK2, lw=0.6)
for ser, ls, lw, col in (('ACE attn', '-', 1.4, INK2), ('ACE mlp', (0, (3, 1.5)), 1.2, INK2), ('ACE attn, even layers only', ':', 1.2, GRAY)):
    d = CV[CV.series == ser]
    lab = {'ACE attn': f"attn: {int(A2.loc['ACE attn', 'observed_n'])} crossing", 'ACE mlp': f"MLP: {int(A2.loc['ACE mlp', 'observed_n'])} crossing",
           'ACE attn, even layers only': 'attn, even layers: 3'}[ser]
    ax.plot(d.g, d.beta, color=col, ls=ls, lw=lw, label=lab)
ax.set_xlabel(r'$g$ (bin centre)'); ax.set_ylabel(r'binned mean $\beta_{\rm layer}$')
ax.legend(loc='upper right', fontsize=5.6, handlelength=1.6, borderaxespad=0.1)
panel_label(ax, 'b')
ax = fig.add_subplot(gs[2])
for s, mk, ls in (('attn', 'o', '-'), ('mlp', 's', (0, (3, 1.5)))):
    d = A[(A.panel == 'c') & (A.channel == s)]
    ax.plot(d.sigma, d.chi_logit, color=INK2, ls=ls, lw=1.1, marker=mk, ms=3.2, mfc='white' if s == 'mlp' else INK2,
            label=f"{'attention' if s == 'attn' else 'MLP'}")
ax.set_yscale('log'); ax.set_xlim(1.03, 0.3); ax.set_xlabel(r'$\sigma$'); ax.set_ylabel('seed variance of logit $g$')
ax.set_title('B9: no peak passes the rule', loc='right', fontsize=6.0, color=INK2, pad=3)
ax.legend(loc='lower right', fontsize=6.0, borderaxespad=0.2)
panel_label(ax, 'c')
save(fig, 'I-F14_ace')
print('I-F14 done')
