"""I-S4  Negative-result atlas (all panels show RETIRED claims; ledger Tier 4 and Round 2).
(a) RETIRED "semantic decoder maximum is the universal C1 time": smoothed semantic Fisher peak vs the A-speed 20% onset
    (normalized time), ledger table "Semantic organization versus flow regime".
(b) RETIRED "trajectory curvature gives a universal C1 detector": normalized time of the maximum turning angle of the
    coarse trajectory per channel (v11 D18 data; includes the early transient).
(c) RETIRED "universal seed collapse": seed share of within-step variance (1 - prompt share, ANOVA) along sigma (R2 E6;
    ZIB on the 9-domain / 428-gen corpus used in Round 2).
Not shown: the passive-susceptibility peak (retired) has no stored per-channel values in the project; see CAPTIONS.md.
Sources: papers/_sources/ledger_v020_text.txt; <SCR>/fig/data/D18_sync_landmark_triangulation.csv; review/round2/out/E6_varcomp_by_step.csv
"""
import io
from common_I import *

TAB = """model,fisher_peak,aspeed_onset20,peak_minus_slowdown
MOCHI,0.138,0.328,-0.190
SD35M,0.692,0.295,0.397
ZIB,0.417,0.438,-0.021
ZIT,0.571,0.786,-0.214"""
ledger_has('Mochi 0.138 0.328 -0.190 0.58 -0.912 SD35M 0.692 0.295 +0.397 1.90 -0.940 ZIB 0.417 0.438 -0.021 2.41 -0.923 ZIT 0.571 0.786 -0.214 1.24 -1.000')
A = pd.read_csv(io.StringIO(TAB)); A['panel'] = 'a'; A['source'] = 'ledger table "Semantic organization versus flow regime" (ledger_v020_text.txt line 354)'
B = pd.read_csv(f'{V11}/D18_sync_landmark_triangulation.csv')[['campaign', 'submodule', 'max_turn_tau', 'max_turn_rad']]
B['panel'] = 'b'; B['source'] = 'v11 fig/data/D18_sync_landmark_triangulation.csv (max-turn columns only)'
E6 = pd.read_csv(f'{R2}/E6_varcomp_by_step.csv'); E6 = E6[E6.ch != 'MOCHI1-attn'].copy()
E6['seed_share'] = 1 - E6.prompt_share_anova; E6['panel'] = 'c'; E6['source'] = 'round2/out/E6_varcomp_by_step.csv (seed share = 1 - prompt_share_anova)'
D = pd.concat([A, B, E6], ignore_index=True)
write_csv(D, 'I-S4_negative_atlas')

fig, axs = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.5), gridspec_kw=dict(width_ratios=[0.9, 0.9, 1.15], wspace=0.5))
ax = axs[0]; y = np.arange(len(A))
for j, r in A.iterrows():
    ax.plot([r.fisher_peak, r.aspeed_onset20], [j, j], color=LIGHT, lw=1.2, zorder=1)
    ax.scatter(r.fisher_peak, j, s=24, color=MODEL_COLOR[r.model], lw=0, zorder=3)
    ax.scatter(r.aspeed_onset20, j, s=22, facecolor='white', edgecolor=MODEL_COLOR[r.model], lw=0.9, zorder=3)
ax.scatter([], [], s=22, color=INK2, label='semantic Fisher peak'); ax.scatter([], [], s=22, facecolor='white', edgecolor=INK2, label='A-speed 20% onset')
ax.set_yticks(y, [SHORT[m] for m in A.model]); ax.set_ylim(len(A) - 0.5, -1.4); ax.set_xlim(0, 1)
ax.set_xlabel('normalized time')
ax.legend(loc='upper left', fontsize=6.2, handletextpad=0.2, borderaxespad=0.05)
panel_label(ax, 'a')

ax = axs[1]
LB = {'Mochi45': 'Mochi', 'SD35M': 'SD3.5 M', 'SD35L': 'SD3.5 L', 'ZIT': 'ZIT', 'ZIB': 'ZIB'}
MC = {'Mochi45': 'MOCHI', 'SD35M': 'SD35M', 'SD35L': 'SD35L', 'ZIT': 'ZIT', 'ZIB': 'ZIB'}
yb = np.arange(len(B))
for j, r in B.reset_index(drop=True).iterrows():
    col = MODEL_COLOR[MC[r.campaign]]
    if r.submodule == 'mlp':
        ax.scatter(r.max_turn_tau, j, s=22, marker='D', facecolor='white', edgecolor=col, lw=0.9, zorder=3)
    else:
        ax.scatter(r.max_turn_tau, j, s=22, color=col, lw=0, zorder=3)
ax.set_yticks(yb, [f"{LB[c]} {'MLP' if s == 'mlp' else 'attn'}" for c, s in zip(B.campaign, B.submodule)], fontsize=7)
ax.set_ylim(len(B) - 0.5, -0.6); ax.set_xlim(0, 1); ax.set_xlabel('time of maximum turn')
panel_label(ax, 'b')

ax = axs[2]
CC = {'ZIT-attn': ('ZIT', '-'), 'ZIT-mlp': ('ZIT', '--'), 'ZIB-attn': ('ZIB', '-'), 'ZIB-mlp': ('ZIB', '--'), 'MOCHI-attn': ('MOCHI', '-'),
      'SD35M-attn': ('SD35M', '-'), 'SD35L-attn': ('SD35L', '-')}
for ch, (m, ls) in CC.items():
    d = E6[E6.ch == ch]
    ax.plot(d.sigma, d.seed_share, color=MODEL_COLOR[m], ls=ls, lw=1.2, label=f"{SHORT[m]} {'MLP' if ch.endswith('mlp') else 'attn'}")
ax.invert_xaxis(); ax.set_xlabel('σ'); ax.set_ylabel('seed share of variance', labelpad=1); ax.set_ylim(0, 0.75)
ax.legend(loc='upper right', fontsize=5.9, ncol=2, handlelength=1.6, columnspacing=0.6, handletextpad=0.3)
panel_label(ax, 'c')
for a_ in axs:
    a_.text(1.0, 1.04, 'RETIRED', transform=a_.transAxes, ha='right', va='bottom', fontsize=7, color=INK, fontweight='bold')
save(fig, 'I-S4_negative_atlas')
print('I-S4 done')
