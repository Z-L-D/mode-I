"""I-S3  Qwen context-clock demotion (register #15-16: Qwen beta ladder DO NOT USE as an anomaly; reported as demoted).
The v1 LLM capture recomputes the full context at every token without a KV cache, so each logged value is a running
statistic of a nested n x D matrix (n = sequence length): monotone in n by construction.
(a,b) Depth-mean g and spectral entropy H of generated tokens against n for Qwen2.5 1.5B / 3B / 7B (median over
      generations in n bins; recomputed from the archive parquet with the R2 E7 definitions).
(c)   The same g normalized by the finite-sample baseline n/(n+D) (g~ of R2 E7) is still exactly monotone in n.
(d)   Median within-generation Spearman rho(n, .) for all seven LLM captures (R2 E7): g, PR/n and g~. Bottom row: the
      model-free reference of R9 B1 (preregistered): nested running statistics of i.i.d. Gaussian rows (D 1536/4096;
      isotropic, power law, power law + spike; 180 simulations) give the same rho with no network at all.
Sources: <SCR>/pq/LLM_qwen25_*.parquet; review/round2/out/E7_llm_context.csv, E7_qwen_beta_ladder.csv;
         review/r9_salvage/out/R9_B1_nested_iid.csv
"""
import glob
from common_I import *

QW = [('qwen25_1_5b', 'Qwen2.5 1.5B'), ('qwen25_3b', 'Qwen2.5 3B'), ('qwen25_7b', 'Qwen2.5 7B')]
QC = [C[0], C[2], C[3]]   # three Qwen sizes: categorical slots in fixed order (no MODEL_COLOR entity for LLMs)
rows = []
for (k, lab) in QW:
    f = glob.glob(f'{SCR}/pq/LLM_{k}_2026*.parquet')[0]
    df = pd.read_parquet(f, columns=['gen_idx', 'token_idx', 'token_type', 'layer', 'g_micro', 'spectral_gap', 'sv_entropy', 'seq_len', 'hidden_dim'])
    df = df[df.token_type == 'generated']; Dm = int(df.hidden_dim.iloc[0]); n = df.seq_len.values
    df = df.assign(gtil=df.g_micro / (n / (n + Dm)))
    m = df.groupby(['gen_idx', 'token_idx']).agg(n=('seq_len', 'first'), g=('g_micro', 'mean'), H=('sv_entropy', 'mean'), gap=('spectral_gap', 'mean'), gtil=('gtil', 'mean')).reset_index()
    bins = np.unique(np.round(np.geomspace(m.n.min(), m.n.quantile(0.95), 25)).astype(int))
    m['nb'] = pd.cut(m.n, bins, include_lowest=True)
    agg = m.groupby('nb', observed=True).agg(n=('n', 'median'), g=('g', 'median'), H=('H', 'median'), gap=('gap', 'median'), gtil=('gtil', 'median'), count=('g', 'size')).reset_index()
    for r in agg.itertuples():
        if r.count >= 50:
            rows.append(dict(panel='abc', model=k, label=lab, D=Dm, n_median=r.n, g_median=r.g, H_median=r.H, gap_median=r.gap, gtil_median=r.gtil, tokens=r.count,
                             source=f'recomputed from pq/{os.path.basename(f)} (generated tokens, depth mean, median per n bin)'))
E7 = pd.read_csv(f'{R2}/E7_llm_context.csv')
for r in E7.itertuples():
    rows.append(dict(panel='d', model=r.model, rho_g=r.rho_g, rho_PRn=r.rho_PRn, rho_gtilde=r.rho_gtilde, gens=r.gens, source='round2/out/E7_llm_context.csv'))
B1 = pd.read_csv(f'{REVIEW}/r9_salvage/out/R9_B1_nested_iid.csv')
rows.append(dict(panel='d', model='nested_iid_null', rho_g=B1.rho_g.median(), rho_PRn=B1.rho_PRn.median(), rho_gtilde=np.nan, gens=len(B1),
                 source=f'r9_salvage/out/R9_B1_nested_iid.csv (median over {len(B1)} simulations; min |rho_g| {B1.rho_g.abs().min():.4f}, min |rho_PRn| {B1.rho_PRn.abs().min():.4f})'))
LAD = pd.read_csv(f'{R2}/E7_qwen_beta_ladder.csv')
for r in LAD.itertuples():
    rows.append(dict(panel='ladder', model=r.model, obs=r.obs, n_cross=r.n_cross, cross_values=r.cross_values, source='round2/out/E7_qwen_beta_ladder.csv'))
D = pd.DataFrame(rows)
write_csv(D, 'I-S3_qwen_context_clock')

fig, axs = plt.subplots(1, 4, figsize=(TEXTWIDTH, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15], wspace=0.62))
for ax, (v, yl, lg) in zip(axs[:3], [('g_median', 'depth-mean g', True), ('H_median', 'depth-mean H', False), ('gtil_median', 'g / [n/(n+D)]', False)]):
    for (k, lab), col in zip(QW, QC):
        d = D[(D.panel == 'abc') & (D.model == k)]
        ax.plot(d.n_median, d[v], color=col, lw=1.3, label=lab.replace('Qwen2.5 ', ''))
    ax.set_xscale('log'); ax.set_xlabel('sequence length n')
    ax.set_xticks([50, 100, 200, 400], ['50', '100', '200', '400']); ax.minorticks_off()
    if lg:
        ax.set_yscale('log'); ax.set_yticks([0.006, 0.01, 0.02, 0.04], ['0.006', '0.01', '0.02', '0.04']); ax.minorticks_off()
    ax.set_ylabel(yl, labelpad=1)
axs[0].legend(loc='upper left', fontsize=6.4, handlelength=1.3, handletextpad=0.3)
for a_, l in zip(axs, 'abcd'): panel_label(a_, l)
ax = axs[3]
p = ax.get_position(); ax.set_position([p.x0 + 0.06, p.y0, p.width - 0.06, p.height])  # room for the row labels
d = D[D.panel == 'd'].reset_index(drop=True); y = np.arange(len(d))
ax.scatter(d.rho_g, y, s=18, color=INK, lw=0, zorder=3, label='g')
ax.scatter(d.rho_PRn, y, s=18, marker='D', facecolor='white', edgecolor=INK, lw=0.8, zorder=3, label='PR/n')
ax.scatter(d.rho_gtilde, y, s=18, marker='s', color=GRAY, lw=0, zorder=3, label='g~')
NAME = {'Llama-3.1-8B-base': 'Llama 8B base', 'llama-31-8B-instruct': 'Llama 8B instr.', 'mistral-03-7B-instruct-ablit': 'Mistral ablit',
        'mistral-03-7B-instruct': 'Mistral instr.', 'qwen25_1_5b': 'Qwen 1.5B', 'qwen25_3b': 'Qwen 3B', 'qwen25_7b': 'Qwen 7B', 'nested_iid_null': 'i.i.d., no model'}
ax.set_yticks(y, [NAME[m] for m in d.model], fontsize=6.4); ax.set_ylim(len(d) - 0.5, -0.6)
ax.set_xlim(-1.45, 1.45); ax.set_xticks([-1, 0, 1]); ax.axvline(0, color=LIGHT, lw=0.8)
ax.axhline(len(d) - 1.5, color=GRAY, lw=0.6, ls=(0, (2, 2)))
ax.set_xlabel('ρ(n, ·) per generation')
ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=6, handletextpad=0.1, borderpad=0.25, handlelength=1.0, framealpha=1)
save(fig, 'I-S3_qwen_context_clock')
print('I-S3 done')
