"""I-F2  Low-dimensional temporal backbone with circular-shift null (register #29: USE WITH WORDING, descriptive).
PC1 variance fraction and effective rank of the population trajectory (after within-state depth standardization) against
999 campaigns in which every time slice is independently circularly shifted in depth (p = 0.001 for every model).
Values transcribed from the ledger card "Low-dimensional temporal backbone in CoS spectral coordinates" (v0.12 table).
Source: papers/_sources/ledger_v020_text.txt (Tier 1 card). Round 1 reproduction: PC1 0.74-0.89, exp(H) 1.53-2.37.
SD1.5 U-Net rows (post hoc, cond branch, attn and MLP): review/r3_unet/out/P1_backbone.csv (same definition and null; the
implementation reproduces the archive SD3.5 M and ZIB attention values to within 0.05, rows marked "calibration" there).
"""
from common_I import *

CARD = 'Tier 1 card "Low-dimensional temporal backbone in CoS spectral coordinates", v0.12 table (ledger_v020_text.txt line 166)'
TAB = """model,pc1,eff_rank,null_pc1_mean,null_rank_mean,p
ZIT,0.836,1.706,0.398,4.88,0.001
ZIB,0.710,2.531,0.351,7.96,0.001
MOCHI,0.874,1.587,0.221,14.84,0.001
SD35M,0.897,1.510,0.188,14.67,0.001
SD35L,0.842,1.784,0.312,9.90,0.001"""
ledger_has('ZIT 0.836 1.706 0.398 4.88 0.001', 'ZIB 0.710 2.531 0.351 7.96 0.001', 'Mochi 0.874 1.587 0.221 14.84 0.001',
           'SD35M 0.897 1.510 0.188 14.67 0.001', 'SD35L 0.842 1.784 0.312 9.90 0.001', 'all five real trajectories beat 999 such campaigns')
import io
D = pd.read_csv(io.StringIO(TAB)); D['source'] = CARD
U = pd.read_csv(f'{REVIEW}/r3_unet/out/P1_backbone.csv'); U = U[U.channel.str.startswith('SD1.5')]
U = pd.DataFrame(dict(model=['SD15attn', 'SD15mlp'], pc1=U.pc1.values, eff_rank=U.eff_rank.values, null_pc1_mean=U.null_pc1_mean.values,
                      null_rank_mean=U.null_rank_mean.values, p=U.p_rank.values, source='review/r3_unet/out/P1_backbone.csv (post hoc, cond)'))
assert (U.p <= 0.001).all()
V = pd.read_csv(f'{REVIEW}/r3_ace/out/P1_backbone.csv')
V = pd.DataFrame(dict(model=['ACEattn', 'ACEmlp'], pc1=V.pc1.values, eff_rank=V.eff_rank.values, null_pc1_mean=V.null_pc1_mean.values,
                      null_rank_mean=V.null_rank_mean.values, p=V.p_rank.values, source='review/r3_ace/out/P1_backbone.csv (post hoc, cond-only, 6 states)'))
D = pd.concat([D, U, V], ignore_index=True)
COL = {**MODEL_COLOR, 'SD15attn': INK, 'SD15mlp': INK, 'ACEattn': INK2, 'ACEmlp': INK2}
LBL = {**SHORT, 'SD15attn': 'SD1.5 attn', 'SD15mlp': 'SD1.5 MLP', 'ACEattn': 'ACE-Step attn', 'ACEmlp': 'ACE-Step MLP'}
D['y'] = np.arange(len(D)) + np.where(D.model.str.startswith(('SD15', 'ACE')), 0.6, 0.0)
write_csv(D, 'I-F2_backbone')

fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH * 0.78, 3.05), gridspec_kw=dict(wspace=0.45))
y = D.y.values
for ax, (o, n, xl, lim) in zip(axs, [('pc1', 'null_pc1_mean', 'PC1 variance fraction', (0, 1)), ('eff_rank', 'null_rank_mean', 'effective rank', (1, 20))]):
    for j, r in D.iterrows():
        ax.plot([r[o], r[n]], [r.y, r.y], color=LIGHT, lw=1.2, zorder=1)
        mk = 's' if r.model == 'SD15mlp' else 'D' if r.model.startswith('ACE') else 'o'
        ax.scatter(r[o], r.y, s=24 if mk != 'o' else 30, marker=mk, color=COL[r.model], facecolor='white' if r.model == 'ACEmlp' else COL[r.model], lw=0.9, zorder=3)
        ax.scatter(r[n], r.y, s=24, facecolor='white', edgecolor=INK2, lw=0.9, zorder=3)
    ax.axhline(4.8, color=GRAY, lw=0.5, ls=':')
    ax.set_yticks(y, [LBL[m] for m in D.model] if o == 'pc1' else [''] * len(D), fontsize=7.5)
    ax.set_ylim(y.max() + 0.5, -1.3); ax.set_xlim(*lim); ax.set_xlabel(xl)
    if o == 'eff_rank':
        ax.set_xscale('log'); ax.set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16']); ax.minorticks_off()
axs[0].scatter([], [], s=30, color=INK2, label='observed'); axs[0].scatter([], [], s=24, facecolor='white', edgecolor=INK2, label='shift-null mean')
axs[0].legend(loc='upper left', ncol=2, fontsize=6.6, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
axs[1].text(19, 5.15, 'U-Net and audio, post hoc', ha='right', va='center', fontsize=6.3, color=INK2, fontstyle='italic')
axs[1].text(19, -0.75, 'transformers (archive)', ha='right', va='center', fontsize=6.3, color=INK2, fontstyle='italic')
panel_label(axs[0], 'a'); panel_label(axs[1], 'b')
save(fig, 'I-F2_backbone')
print('I-F2 done')
