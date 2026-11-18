"""I-F3  Observable hierarchy and lineage separation.
(a) ZIT<->ZIB (lineage) vs SD3.5 M<->L (same family) vs the 8 unrelated pairs at five observable levels (R6 S1a-c).
(b) Base<->Turbo tangent cosine under every capture functional (R2 E10, R5 Q2) against the best unrelated pair.
Sources: review/round6/out/S1_hierarchy.csv, S1_results.json; review/round2/out/E10_tangent.csv; review/round5/out/Q_results.json
"""
from common_I import *

H = pd.read_csv(f'{R6}/S1_hierarchy.csv')
LV = ['L0_D1', 'L0_D2', 'L1_field', 'L2_shape', 'L3_tangent']
LAB = ['g\nenvelope', 'log g\nenvelope', 'depth×σ\nfield', 'registered\nshape', 'tangent']
rows = []
for i, k in enumerate(LV):
    for r in H.itertuples():
        rows.append(dict(panel='a', level=k, a=r.a, b=r.b, kind=r.kind, value=getattr(r, k)))
A = pd.DataFrame(rows)
marg = {k: A[(A.level == k) & (A.kind == 'lineage')].value.iloc[0] - A[(A.level == k) & (A.kind == 'unrelated')].value.max() for k in LV}
S1 = rjson(f'{R6}/S1_results.json')['margin']
for k in LV:
    assert abs(marg[k] - S1[k]) < 1e-9, k

# (b) lineage tangent by capture functional
E10 = pd.read_csv(f'{R2}/E10_tangent.csv').set_index(['a', 'b'])
Q = rjson(f'{R5}/Q_results.json')
B = [
    ('Archive (ZIT-25 pooled)', E10.loc[('ZIT-attn', 'ZIB-attn'), 'tangent_cos'], E10.loc[('ZIT-mlp', 'ZIB-mlp'), 'tangent_cos'], 'round2/out/E10_tangent.csv'),
    ('CFG 4 uncond', Q['Q2b_tangent_ZIB4uncond_ZIT_attn'], Q['Q2b_tangent_ZIB4uncond_ZIT_mlp'], 'round5/out/Q_results.json Q2b'),
    ('CFG 4 branch mean', Q['Q2_ref_tangent_ZIB4BMsynth_ZIT_attn'], Q['Q2_ref_tangent_ZIB4BMsynth_ZIT_mlp'], 'round5/out/Q_results.json Q2_ref'),
    ('CFG 4 cond', Q['Q2a_tangent_ZIB4cond_ZIT_attn'], Q['Q2a_tangent_ZIB4cond_ZIT_mlp'], 'round5/out/Q_results.json Q2a'),
    ('CFG 1 cond', Q['Q2_ref_tangent_ZIB1new_ZIT_attn'], Q['Q2_ref_tangent_ZIB1new_ZIT_mlp'], 'round5/out/Q_results.json Q2_ref'),
]
unrel_max = H[H.kind == 'unrelated'].L3_tangent.max()
brows = []
for lab, at, ml, src in B:
    brows.append(dict(panel='b', level='L3_tangent', a='ZIT', b=lab, kind='lineage attn', value=at, source=src))
    brows.append(dict(panel='b', level='L3_tangent', a='ZIT', b=lab, kind='lineage mlp', value=ml, source=src))
brows.append(dict(panel='b', level='L3_tangent', a='best unrelated', b='ZIT-25–Mochi (attn)', kind='unrelated max', value=unrel_max, source='round6/out/S1_hierarchy.csv'))
A['source'] = 'round6/out/S1_hierarchy.csv'
out = pd.concat([A, pd.DataFrame(brows)], ignore_index=True)
write_csv(out, 'I-F3_observable_hierarchy')

fig = plt.figure(figsize=(TEXTWIDTH, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
for i, k in enumerate(LV):
    u = A[(A.level == k) & (A.kind == 'unrelated')].value.values
    ax.scatter(i + np.linspace(-0.16, 0.16, len(u)), u, s=14, color=GRAY, zorder=3, lw=0, label='Unrelated pairs (8)' if i == 0 else None)
    sf = A[(A.level == k) & (A.kind == 'same family')].value.iloc[0]
    ax.scatter(i + 0.26, sf, s=20, marker='s', color=MODEL_COLOR['SD35M'], zorder=3, lw=0, label='SD3.5 M–L (same family)' if i == 0 else None)
    l = A[(A.level == k) & (A.kind == 'lineage')].value.iloc[0]
    ax.scatter(i, l, s=34, color=MODEL_COLOR['ZIT'], zorder=4, edgecolor='white', lw=0.8, label='ZIT–ZIT-25 (same weights)' if i == 0 else None)
    ax.text(i, 1.13, f'{marg[k]:+.2f}', ha='center', va='bottom', fontsize=7.5, color=INK)
ax.text(-0.62, 1.13, 'margin', ha='right', va='bottom', fontsize=7, color=INK2)
ax.set_xticks(range(len(LV)), LAB, fontsize=6.6)
ax.set_xlim(-0.6, len(LV) - 0.4); ax.set_ylim(-0.4, 1.12)
ax.axhline(0, color=LIGHT, lw=0.8, zorder=1)
ax.set_ylabel('Similarity (Pearson r or cosine)')
ax.legend(loc='lower left', fontsize=6.8, handletextpad=0.3, borderaxespad=0.2)
panel_label(ax, 'a')

ax = fig.add_subplot(gs[1])
y = np.arange(len(B))
for j, (lab, at, ml, _) in enumerate(B):
    ax.plot([min(at, ml), max(at, ml)], [j, j], color=LIGHT, lw=1.2, zorder=2)
    ax.scatter(at, j, s=22, color=C[0], zorder=3, lw=0, label='Attention' if j == 0 else None)
    ax.scatter(ml, j, s=22, marker='D', color=C[2], zorder=3, lw=0, label='MLP' if j == 0 else None)
ax.set_yticks(y, [b[0] for b in B], fontsize=7)
ax.invert_yaxis(); ax.set_xlim(0.93, 1.002)
ax.set_xlabel('ZIT–ZIT-25 tangent cosine (same weights)')
ax.text(0.932, len(B) - 0.45, f'best unrelated pair: {unrel_max:.2f}\n(off scale)', fontsize=6.6, color=INK2, va='center')
ax.set_ylim(len(B) + 0.1, -0.6)
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.55), fontsize=6.8, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax, 'b')
save(fig, 'I-F3_observable_hierarchy')
print('I-F3 done', {k: round(v, 3) for k, v in marg.items()})
