"""I-F15  Three tiers (with the Round 16 and 17 controls) of dynamical alignment, the layer-axis crossing census, and the
ACE-Step Base vs Turbo depth profiles (Rounds 12 and 14; preregistered values, plotted without refitting).
(a) Matched-sigma tangent cosine (Round 6 S1 estimator, unchanged) for same-weights pairs, distillation-lineage pairs
    (Z-Image Base vs Turbo, R12; ACE-Step Base vs Turbo, R14) and unrelated pairs, attention (filled) and MLP (open), with
    each lineage pair's layer-permutation null 95th percentile (gray ticks).
(b) Share of 100 prompt bootstraps with 1, 2, 3 or >= 4 layer-axis crossings of the analyzer's beta_layer(g), all steps
    pooled. Z-Image cubes: R12-8 (r12_8_crossings.csv); ACE-Step Base: R12-7 / B6 replay (r14_B6_boot.csv) and the post hoc
    MLP replay (posthoc_aceb_crossings.csv); ACE-Step Turbo: R3 A2 (r3_ace/out/A2_bootstrap.csv); SD3.5 and Mochi
    archive: R6 A2 (round6/out/A2_beta_layer_bootstrap.csv; only the 1, 3 and >= 5 shares are recorded there, the rest
    is shown as "other").
(c) Layer profile of mean log g, Base at the Turbo sigmas (R14-2 detail), attention and MLP.
Sources: review/zib_turbo_mixup_2026-10-18/r12/out/{r12_tangents.csv, r12_unrelated_attn.json, r12_8_crossings.csv};
         review/ace_base_r14/out/{r14_tangents.csv, posthoc_ace_vs_image_tangents.csv, r14_B6_boot.csv,
         posthoc_aceb_crossings.csv, r14_details_A.json}; review/round2/out/E10_tangent.csv; review/r3_ace/out/A2_bootstrap.csv;
         review/round6/out/A2_beta_layer_bootstrap.csv.
"""
import ast, json
from common_I import *

R12 = f'{REVIEW}/zib_turbo_mixup_2026-10-18/r12/out'; R14 = f'{REVIEW}/ace_base_r14/out'
R16 = f'{REVIEW}/ace_turbo_panel_r16/out'; R17 = f'{REVIEW}/tier_controls_r17/out'
T16 = pd.read_csv(f'{R16}/r16_tangents.csv').drop_duplicates(['a', 'b', 'sub']).set_index(['a', 'b', 'sub'])
T17 = pd.read_csv(f'{R17}/r17_tangents.csv').drop_duplicates(['a', 'b', 'sub']).set_index(['a', 'b', 'sub'])
T12 = pd.read_csv(f'{R12}/r12_tangents.csv').set_index(['a', 'b', 'sub'])
T14 = pd.read_csv(f'{R14}/r14_tangents.csv').set_index(['a', 'b', 'sub'])
E10 = pd.read_csv(f'{R2}/E10_tangent.csv').set_index(['a', 'b'])
UN12 = json.load(open(f'{R12}/r12_unrelated_attn.json'))
PA = pd.read_csv(f'{R14}/posthoc_ace_vs_image_tangents.csv')

rows = []
def add(tier, label, at, ml, p95a=np.nan, p95m=np.nan, src=''):
    rows.append(dict(panel='a', tier=tier, pair=label, tangent_attn=at, tangent_mlp=ml, null_p95_attn=p95a, null_p95_mlp=p95m, source=src))

# same weights
add('same weights', 'ZIT 8 vs 25 steps', E10.loc[('ZIT-attn', 'ZIB-attn'), 'tangent_cos'], E10.loc[('ZIT-mlp', 'ZIB-mlp'), 'tangent_cos'], src='round2/out/E10_tangent.csv')
add('same weights', 'Z-Image Turbo, matched rerun (Z3 vs ZIT)', T12.loc[('Z3', 'ZIT', 'attn'), 'tangent'], T12.loc[('Z3', 'ZIT', 'mlp'), 'tangent'], src='r12_tangents.csv')
add('same weights', 'Z-Image Base cond vs uncond', T12.loc[('Z1', 'Z1U', 'attn'), 'tangent'], T12.loc[('Z1', 'Z1U', 'mlp'), 'tangent'], src='r12_tangents.csv')
add('same weights', 'ACE-Step Base GPU 0 vs 1', T14.loc[('ACEB0', 'ACEB1', 'attn'), 'tangent'], T14.loc[('ACEB0', 'ACEB1', 'mlp'), 'tangent'], src='r14_tangents.csv')
add('same weights', 'ACE-Step Base cond vs uncond', T14.loc[('ACEB', 'ACEBU', 'attn'), 'tangent'], T14.loc[('ACEB', 'ACEBU', 'mlp'), 'tangent'], src='r14_tangents.csv')
add('same weights', 'ACE-Step Turbo 8 vs 6 steps, two capture sessions (R16)', T16.loc[('T8', 'ACET', 'attn'), 'tangent'], T16.loc[('T8', 'ACET', 'mlp'), 'tangent'], src='r16_tangents.csv (R16-1)')
for k in (3, 4, 5, 7, 9, 10):
    add('same weights', f'ACE-Step Turbo {k} vs 8 steps (R16)', T16.loc[(f'T{k}', 'T8', 'attn'), 'tangent'], T16.loc[(f'T{k}', 'T8', 'mlp'), 'tangent'], src='r16_tangents.csv (R16-4)')
add('same weights', 'SD3.5 M recapture vs Round 11 (R17)', T17.loc[('SD35M', 'SD3SV', 'attn'), 'tangent'], np.nan, src='r17_tangents.csv (R17-3)')
for n, lab in [('PIXART', 'PixArt-Sigma'), ('COSMOS2B', 'Cosmos-Predict2 2B')]:
    add('same weights', f'{lab} cond vs uncond (R17)', T17.loc[(n, n + 'U', 'attn'), 'tangent'], T17.loc[(n, n + 'U', 'mlp'), 'tangent'], src='r17_tangents.csv (R17-3)')
for n, lab in [('SD3M', 'SD3 Medium'), ('SD35M', 'SD3.5 Medium')]:
    add('same weights', f'{lab} cond vs uncond (R17)', T17.loc[(n, n + 'U', 'attn'), 'tangent'], np.nan, src='r17_tangents.csv (R17-3)')
# lineage (distillation pairs and the same-vendor successor pair)
add('lineage', 'Z-Image Base vs Turbo (Z1 vs ZIT)', T12.loc[('Z1', 'ZIT', 'attn'), 'tangent'], T12.loc[('Z1', 'ZIT', 'mlp'), 'tangent'],
    T12.loc[('Z1', 'ZIT', 'attn'), 'null_p95'], T12.loc[('Z1', 'ZIT', 'mlp'), 'null_p95'], 'r12_tangents.csv (R12-1)')
add('lineage', 'Z-Image Base vs matched Turbo (Z1 vs Z3)', T12.loc[('Z1', 'Z3', 'attn'), 'tangent'], T12.loc[('Z1', 'Z3', 'mlp'), 'tangent'],
    T12.loc[('Z1', 'Z3', 'attn'), 'null_p95'], T12.loc[('Z1', 'Z3', 'mlp'), 'null_p95'], 'r12_tangents.csv')
add('lineage', 'ACE-Step Base vs Turbo', T14.loc[('ACEB', 'ACET', 'attn'), 'tangent'], T14.loc[('ACEB', 'ACET', 'mlp'), 'tangent'],
    T14.loc[('ACEB', 'ACET', 'attn'), 'null_p95'], T14.loc[('ACEB', 'ACET', 'mlp'), 'null_p95'], 'r14_tangents.csv (R12-6)')
add('lineage', 'ACE-Step Base vs Turbo at 8 steps (R16)', T16.loc[('ACEB', 'T8', 'attn'), 'tangent'], T16.loc[('ACEB', 'T8', 'mlp'), 'tangent'],
    T16.loc[('ACEB', 'T8', 'attn'), 'null_p95'], T16.loc[('ACEB', 'T8', 'mlp'), 'null_p95'], 'r16_tangents.csv (R16-2)')
add('lineage', 'SD3 Medium vs SD3.5 Medium, same vendor, no distillation (R17)', T17.loc[('SD3M', 'SD35M', 'attn'), 'tangent'], np.nan,
    T17.loc[('SD3M', 'SD35M', 'attn'), 'null_p95'], np.nan, 'r17_tangents.csv (R17-2)')
# unrelated: Z-Image Base/Turbo vs other image DiTs (attn, R12) and ACE vs image models (attn and MLP, R14 post hoc)
for k, v in UN12.items():
    a, b = k.split('|'); add('unrelated', f'{a} vs {b}', v, np.nan, src='r12_unrelated_attn.json')
for r in PA.itertuples():
    add('unrelated', f'{r.a} vs {r.b}', r.tangent if r.sub == 'attn' else np.nan, r.tangent if r.sub == 'mlp' else np.nan, src='posthoc_ace_vs_image_tangents.csv (post hoc)')

add('unrelated', 'PixArt-Sigma vs Cosmos-Predict2 2B, 28 layers each (R17)', T17.loc[('PIXART', 'COSMOS2B', 'attn'), 'tangent'], T17.loc[('PIXART', 'COSMOS2B', 'mlp'), 'tangent'],
    T17.loc[('PIXART', 'COSMOS2B', 'attn'), 'null_p95'], T17.loc[('PIXART', 'COSMOS2B', 'mlp'), 'null_p95'], 'r17_tangents.csv (R17-1)')

# (b) crossing census
def dist_row(label, dist, src):
    d = {int(k): v / 100 for k, v in dist.items()}
    rows.append(dict(panel='b', channel=label, c1=d.get(1, 0), c2=d.get(2, 0), c3=d.get(3, 0), c4plus=sum(v for k, v in d.items() if k >= 4), other=0.0, source=src))
C8 = pd.read_csv(f'{R12}/r12_8_crossings.csv').set_index('cube')
R6A = pd.read_csv(f'{R6}/A2_beta_layer_bootstrap.csv'); R3A = pd.read_csv(f'{REVIEW}/r3_ace/out/A2_bootstrap.csv').set_index('channel')
PC = pd.read_csv(f'{R14}/posthoc_aceb_crossings.csv').set_index('cube'); B6 = pd.read_csv(f'{R14}/r14_B6_boot.csv')
for nm, lab in [('SD35M archive', 'SD3.5 M attn'), ('SD35L archive', 'SD3.5 L attn'), ('MOCHI archive CFG4.5', 'Mochi attn')]:
    r = R6A[(R6A.set == nm) & (R6A.submodule == 'as loaded (pooled)')].iloc[0]
    rows.append(dict(panel='b', channel=lab, c1=r.frac_1, c2=0.0, c3=r.frac_3, c4plus=r.frac_5plus, other=max(0.0, 1 - r.frac_1 - r.frac_3 - r.frac_5plus),
                     source='round6/out/A2_beta_layer_bootstrap.csv'))
for cube_, lab in [('ZIB_attn', 'ZIT-25 attn'), ('ZIB_mlp', 'ZIT-25 MLP'), ('ZIT_attn', 'ZIT attn'), ('ZIT_mlp', 'ZIT MLP'),
                   ('Z3_attn', 'ZIT matched attn'), ('Z3_mlp', 'ZIT matched MLP'), ('Z1_attn', 'Z-Image Base attn'), ('Z1_mlp', 'Z-Image Base MLP')]:
    dist_row(lab, ast.literal_eval(C8.loc[cube_, 'count_dist']), 'r12_8_crossings.csv (R12-8)')
for ch, lab in [('ACE attn', 'ACE-Step Turbo attn'), ('ACE mlp', 'ACE-Step Turbo MLP')]:
    r = R3A.loc[ch]
    rows.append(dict(panel='b', channel=lab, c1=r.frac_1, c2=0.0, c3=r.frac_3, c4plus=r.frac_5plus, other=max(0.0, 1 - r.frac_1 - r.frac_3 - r.frac_5plus), source='r3_ace/out/A2_bootstrap.csv'))
dist_row('ACE-Step Turbo 8 st. attn', pd.Series(json.load(open(f'{R16}/r16_details.json'))['B6_counts']).value_counts().to_dict(), 'r16_details.json (R16-5)')
dist_row('ACE-Step Base attn', B6.n_cross.value_counts().to_dict(), 'r14_B6_boot.csv (R12-7)')
dist_row('ACE-Step Base MLP', ast.literal_eval(PC.loc['ACEB_mlp', 'count_dist']), 'posthoc_aceb_crossings.csv (post hoc)')

# (c) depth profiles
DA = json.load(open(f'{R14}/r14_details_A.json'))['R14_2']
for sub in ('attn', 'mlp'):
    for l, (b, t) in enumerate(zip(DA[sub]['base'], DA[sub]['turbo'])):
        rows.append(dict(panel='c', sub=sub, layer=l, logg_base=b, logg_turbo=t, source='r14_details_A.json R14_2'))
D = pd.DataFrame(rows)
write_csv(D, 'I-F15_tiers')

fig = plt.figure(figsize=(TEXTWIDTH, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.9, 0.95], wspace=0.95)
ax = fig.add_subplot(gs[0])
a = D[D.panel == 'a']; TIERS = [('same weights', INK, 2), ('lineage', MODEL_COLOR['ZIT'], 1), ('unrelated', GRAY, 0)]
rng = np.random.default_rng(0)
for tier, col, y in TIERS:
    d = a[a.tier == tier]
    ja = rng.uniform(-0.17, 0.17, len(d)); jm = rng.uniform(-0.17, 0.17, len(d))
    ax.scatter(d.tangent_attn, y + 0.12 + ja * 0.5, s=16, color=col, lw=0, zorder=3)
    ax.scatter(d.tangent_mlp, y - 0.18 + jm * 0.5, s=16, facecolor='white', edgecolor=col, lw=0.9, zorder=3)
    for r in d.itertuples():
        for v, yy in ((r.null_p95_attn, y + 0.12), (r.null_p95_mlp, y - 0.18)):
            if np.isfinite(v): ax.plot([v, v], [yy - 0.13, yy + 0.13], color=GRAY, lw=1.0, zorder=2)
for r in a[a.tier == 'lineage'].itertuples():
    if r.pair.startswith('ACE-Step Base vs Turbo at 8'):
        ax.annotate('ACE', (r.tangent_mlp, 0.82), xytext=(0, -9), textcoords='offset points', fontsize=6.2, ha='center', color=INK2)
    if r.pair.startswith('SD3 Medium'):
        ax.annotate('SD3', (r.tangent_attn, 1.12), xytext=(0, 6), textcoords='offset points', fontsize=6.2, ha='center', color=INK2)
for r in a[a.pair.str.startswith('PixArt-Sigma vs')].itertuples():
    ax.annotate('PixArt–Cosmos', (r.tangent_mlp, -0.18), xytext=(2, -11), textcoords='offset points', fontsize=6.2, ha='right', color=INK2)
ax.axvline(0, color=LIGHT, lw=0.8, zorder=1)
ax.set_yticks([2, 1, 0], ['same\nweights', 'lineage', 'unrelated'], fontsize=7); ax.set_ylim(-0.6, 2.6); ax.set_xlim(-0.3, 1.05)
ax.set_xlabel('tangent cosine')
ax.scatter([], [], s=16, color=INK2, label='attn'); ax.scatter([], [], s=16, facecolor='white', edgecolor=INK2, label='MLP')
ax.plot([], [], color=GRAY, lw=1.0, label='null p95')
ax.legend(loc='upper left', fontsize=6.0, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax, 'a')

ax = fig.add_subplot(gs[1])
b = D[D.panel == 'b'].reset_index(drop=True); y = np.arange(len(b)); left = np.zeros(len(b))
for k, col, lab in [('c1', GRAY, '1'), ('c2', C[2], '2'), ('c3', C[0], '3'), ('c4plus', C[3], '≥ 4'), ('other', LIGHT, 'other')]:
    ax.barh(y, b[k], left=left, color=col, height=0.72, lw=0, label=lab); left += b[k].values
ax.set_yticks(y, b.channel, fontsize=6.0); ax.set_ylim(len(b) - 0.5, -0.6); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
ax.set_xlabel('bootstrap share by crossings')
ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.17), ncol=5, fontsize=6.0, handlelength=0.9, columnspacing=0.6, handletextpad=0.3)
ax.grid(axis='y', visible=False)
panel_label(ax, 'b')

ax = fig.add_subplot(gs[2])
c = D[D.panel == 'c']
for sub, ls in (('attn', '-'), ('mlp', '--')):
    d = c[c['sub'] == sub]
    ax.plot(d.layer, d.logg_turbo, color=C[0], ls=ls, lw=1.2, label=f'Turbo {"attn" if sub == "attn" else "MLP"}')
    ax.plot(d.layer, d.logg_base, color=C[1], ls=ls, lw=1.2, label=f'Base {"attn" if sub == "attn" else "MLP"}')
ax.set_xlabel('ACE-Step layer'); ax.set_ylabel(r'mean $\ln g$', labelpad=1); ax.set_xticks(range(0, 24, 6))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.0, handlelength=1.6, columnspacing=0.8, borderaxespad=0.1)
panel_label(ax, 'c')
save(fig, 'I-F15_tiers')
print('I-F15 done', len(D))
