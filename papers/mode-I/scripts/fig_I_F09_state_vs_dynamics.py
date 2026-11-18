"""I-F9  State alignment versus dynamical alignment (register #21; ledger D10 card).
(a) Registered shape r (state) against the matched-sigma tangent cosine (dynamics) for all 10 model pairs (R6 S1, attention).
(b) Ledger D10: state-only alignment maps (raw, Procrustes, linear, CCA; learned from state landmarks without derivative
    labels) raise the matched state correlation of unrelated population fields, while the matched depth-derivative
    correlation stays near zero; Base<->Turbo is about 0.94 under every map.
Sources: review/round6/out/S1_hierarchy.csv; D10_common_latent_alignment.csv from the v11 figure-data package
(<SCR>/fig/data/, reproducer D10_common_latent_alignment.py), quoted in the ledger card
"Static representation alignment does not rescue cross-architecture dynamics".
"""
from common_I import *

ledger_has('matched derivative correlation averages about 0.937', 'best mapping for each unrelated pair averages only about 0.007')
H = pd.read_csv(f'{R6}/S1_hierarchy.csv')
D10 = pd.read_csv(f'{V11}/D10_common_latent_alignment.csv')
D10['pair'] = D10.apply(lambda r: '–'.join(sorted([r.source, r.target])) + ' ' + r.submodule, axis=1)
D10['cls'] = np.where(D10.source.isin(['ZIT', 'ZIB']) & D10.target.isin(['ZIT', 'ZIB']), 'lineage',
                      np.where(D10.source.str.startswith('SD35') & D10.target.str.startswith('SD35'), 'same family', 'unrelated'))
lin_mean = D10[D10.cls == 'lineage'].matched_derivative_r.mean()
unrel_best = D10[D10.cls != 'lineage'].groupby(['source', 'target']).matched_derivative_r.max().mean()
assert abs(lin_mean - 0.937) < 0.0005 and abs(unrel_best - 0.007) < 0.0005
rows = [dict(panel='a', a=r.a, b=r.b, kind=r.kind, state_shape_r=r.L2_shape, field_r=r.L1_field, tangent_cos=r.L3_tangent, source='round6/out/S1_hierarchy.csv')
        for r in H.itertuples()]
rows += [dict(panel='b', a=r.source, b=r.target, kind=r.cls, submodule=r.submodule, alignment=r.alignment, matched_state_r=r.matched_state_r,
              matched_derivative_r=r.matched_derivative_r, source='v11 fig/data/D10_common_latent_alignment.csv (ledger D10)') for r in D10.itertuples()]
D = pd.DataFrame(rows)
write_csv(D, 'I-F9_state_vs_dynamics')

KC = {'lineage': MODEL_COLOR['ZIT'], 'same family': MODEL_COLOR['SD35M'], 'unrelated': GRAY}
fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
a = D[D.panel == 'a']
for k, mk in [('unrelated', 'o'), ('same family', 's'), ('lineage', 'o')]:
    d = a[a.kind == k]
    ax.scatter(d.state_shape_r, d.tangent_cos, s=26 if k == 'lineage' else 18, marker=mk, color=KC[k], lw=0, zorder=3,
               label={'lineage': 'ZIT–ZIT-25 (same weights)', 'same family': 'SD3.5 M–L', 'unrelated': 'unrelated pairs'}[k])
for r in a.itertuples():
    if r.kind == 'unrelated' and 'MOCHI' in (r.a, r.b) and r.a in ('ZIT', 'ZIB'):
        off = (4, 7) if r.a == 'ZIB' else (4, -10)
        ax.annotate(pair_label(r.a, r.b), (r.state_shape_r, r.tangent_cos), xytext=off, textcoords='offset points', fontsize=6.3, color=INK2, ha='center')
ax.axhline(0, color=LIGHT, lw=0.8, zorder=1)
ax.set_xlim(-0.15, 1.05); ax.set_ylim(-0.35, 1.05)
ax.set_xlabel('registered shape r (state)'); ax.set_ylabel('tangent cosine (dynamics)')
ax.legend(loc='upper left', fontsize=6.6, handletextpad=0.2)
panel_label(ax, 'a')

ax = axs[1]
b = D[D.panel == 'b']
MAPS = [('raw', 'o', 'raw'), ('procrustes', 's', 'Procrustes'), ('linear', 'D', 'linear'), ('cca', '^', 'CCA')]
for (src, tgt, sub), d in b.groupby(['a', 'b', 'submodule']):
    raw = d[d.alignment == 'raw'].iloc[0]; best = d.loc[d.matched_state_r.idxmax()]
    ax.annotate('', xy=(best.matched_state_r, best.matched_derivative_r), xytext=(raw.matched_state_r, raw.matched_derivative_r),
                arrowprops=dict(arrowstyle='->', color=KC[raw.kind], lw=0.7, shrinkA=2, shrinkB=2), zorder=2)
for al, mk, lb in MAPS:
    d = b[b.alignment == al]
    ax.scatter(d.matched_state_r, d.matched_derivative_r, s=13, marker=mk, c=[KC[k] for k in d.kind], lw=0, zorder=3)
    ax.scatter([], [], s=13, marker=mk, color=INK2, label=lb)
ax.axhline(0, color=LIGHT, lw=0.8, zorder=1)
ax.text(0.955, 0.86, 'ZIT–ZIT-25\nattn, MLP', fontsize=6.5, color=INK, ha='right', va='top')
ax.text(0.2, -0.24, 'SD3.5 M–L', fontsize=6.5, color=INK, ha='left')
ax.text(0.62, -0.30, 'ZIT–Mochi,\nSD3.5 M–Mochi', fontsize=6.5, color=INK, ha='center', va='center')
ax.set_xlim(0.15, 1.02); ax.set_ylim(-0.4, 1.05)
ax.set_xlabel('matched state r after alignment'); ax.set_ylabel('matched depth-derivative r')
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.6), fontsize=6.4, handletextpad=0.2, title='state-only map', title_fontsize=6.4)
panel_label(ax, 'b')
save(fig, 'I-F9_state_vs_dynamics')
print('I-F9 done', round(lin_mean, 3), round(unrel_best, 4))
