"""I-F1  Corpus and instrument map (register M1-M4 methods facts; M2 wording: name the capture regime for every number).
Each row is one campaign. The strip shows, along normalized sampling progress, which capture functional the CSV value
records (ComfyUI CFG batching decides it): cond-only, pooled cond+uncond, branch mean, or explicit per-branch rows
(instrument v2). Right-hand columns: steps, layers, hooked submodule, CFG, generations / prompts.
Diffusion counts are read from the cubes; regimes, CFG and the AR rows are transcribed from the ledger archive table and
capture-instrument audit (papers/_sources/ledger_v020_text.txt) and the v11 architecture inventory (layers, hidden dim).
"""
import pickle, io
from matplotlib.patches import Rectangle
from common_I import *

# ledger_has('cond-only, seq 4160 (4096 image + 64 text)', 'pooled cond+uncond, seq 8320', 'branch-mean, seq 11130',
#            'step 0 pooled (8192); step 1 mixed (38% pooled); steps 2+ branch-mean (4096)', 'pooled in 74% of steps, by domain (decodable at 1.000); rest branch-mean',
#            'v1: no KV cache → nested running statistic', '60 prompts × 1 cond-only')


def cnt_arch(name):
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{name}.pkl', 'rb'))
    return X.shape[1], X.shape[2], len(X), meta.pt.astype(str).nunique()


def cnt_r5(name):
    d = pickle.load(open(f'{SCR}/R5_{name}.pkl', 'rb')); X = d['X'][list(d['X'])[0]]
    return X.shape[1], X.shape[2], len(X), d['meta'].pt.astype(str).nunique(), list(d['X'])


def cnt_r3(name):
    g, meta, X, s = pickle.load(open(f'{SCR}/F3_{name}.pkl', 'rb'))
    return X.shape[1], X.shape[2], len(X), meta.pt.astype(str).nunique()


rows = []
SRC_L = 'ledger archive table + capture-instrument audit'
# (group, label, submodule, cfg, steps, layers, gens, prompts, regime segments [(start_frac, end_frac, regime)], note, source)
for m, lab, sub, cfg, reg, note, _cube in [
        ('ZIT', 'Z-Image Turbo', 'attn+MLP', '1.0', [(0, 1, 'cond-only')], 'text rows in X', 'F_ZIT_attn.pkl'),
        ('ZIB', 'Z-Image Base', 'attn+MLP', '4.0', [(0, 1, 'pooled')], '', 'F_ZIB_attn.pkl'),
        ('MOCHI', 'Mochi 1', 'attn', '4.5', [(0, 1, 'branch mean')], 'header-less file', 'F_MOCHI_attn.pkl'),
        ('MOCHI1', 'Mochi 1', 'attn', '1.0', [(0, 1, 'cond-only')], 'CFG + steps arm', 'F_MOCHI1_attn.pkl'),
        ('SD35M', 'SD3.5 Medium', 'attn', '4.5', None, '', 'F_SD35M_attn.pkl'),
        ('SD35L', 'SD3.5 Large', 'attn', '3.5', [(0, 1, 'by domain')], 'pooled 74% of steps', 'F_SD35L_attn.pkl')]:
    T, L, G, P = cnt_arch(f'{m}_attn')
    if reg is None:  # SD3.5 M: step 0 pooled, step 1 mixed, steps 2+ branch mean
        reg = [(0, 1 / (T - 1), 'pooled'), (1 / (T - 1), 2 / (T - 1), 'mixed'), (2 / (T - 1), 1, 'branch mean')]
    rows.append(dict(group='Diffusion archive (passive guider v1)', label=lab, submodule=sub, cfg=cfg, steps=T, layers=L, gens=G, prompts=P, regime=reg, note=note,
                     source=f'counts: cube F_{m}_attn.pkl; regime/CFG: {SRC_L}'))
T, L, G, P, br = cnt_r5('Z40_1024_s1001_attn')
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='Z-Image Base', submodule='attn+MLP', cfg='4.0', steps=T, layers=L, gens=G, prompts=P,
                 regime=[(0, 1, 'explicit')], note='cond + uncond', source=f'counts: cube R5_Z40_1024_s1001_attn.pkl ({"/".join(br)})'))
T, L, G, P, br = cnt_r5('Z10_1024_s1001_attn'); T3, L3, G3, P3 = cnt_r3('Z10_1024_s1001_attn')
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='Z-Image Base', submodule='attn+MLP', cfg='1.0', steps=T, layers=L, gens=G + G3,
                 prompts=P + P3, regime=[(0, 1, 'cond-only')], note='R4 20 + R3 60', source='counts: cubes R5_Z10_1024_s1001_attn.pkl, F3_Z10_1024_s1001_attn.pkl'))
g45 = [cnt_r5(f'S45_{r}_s100{k}_attn') for r, k in [('1024', 1), ('1024', 2), ('768', 1), ('512', 1)]]
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='SD3.5 Medium', submodule='attn', cfg='4.5', steps=g45[0][0], layers=g45[0][1],
                 gens=sum(x[2] for x in g45), prompts=g45[0][3], regime=[(0, 1, 'explicit')], note='512/768/1024 px', source='counts: cubes R5_S45_{1024,768,512}_s100*_attn.pkl'))
g10 = [cnt_r5(f'S10_1024_s100{k}_attn') for k in (1, 2)]
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='SD3.5 Medium', submodule='attn', cfg='1.0', steps=g10[0][0], layers=g10[0][1],
                 gens=sum(x[2] for x in g10), prompts=g10[0][3], regime=[(0, 1, 'cond-only')], note='seeds 1001/1002', source='counts: cubes R5_S10_1024_s100*_attn.pkl'))
pyr = {k: cnt_r5(k + '_attn') for k in ['P10_40', 'P10_100', 'P45_40', 'P45_100']}
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='SD3.5 M step-count pair', submodule='attn', cfg='1.0 / 4.5', steps='40 / 100',
                 layers=pyr['P10_40'][1], gens=sum(v[2] for v in pyr.values()), prompts=1, regime=[(0, 1, 'explicit')], note='one prompt',
                 source='counts: cubes R5_P10_40/100, R5_P45_40/100'))
UZ = np.load(f'{REVIEW}/r3_unet/out/u0_cache.npz', allow_pickle=True)
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='SD1.5 U-Net (R3)', submodule='attn+MLP', cfg='7.5',
                 steps=UZ['attn_cond_g_micro'].shape[1], layers=UZ['attn_cond_g_micro'].shape[2], gens=len(UZ['gids']), prompts=len(np.unique(UZ['pt'])),
                 regime=[(0, 1, 'explicit')], note='512 px; transformer blocks', source='counts: review/r3_unet/out/u0_cache.npz (R3 nodes.py, UNetModel entry)'))
AZ = np.load(f'{REVIEW}/r3_ace/out/cache_attn.npz', allow_pickle=True)
rows.append(dict(group='Explicit-branch controls (instrument v2, R4) and R3', label='ACE-Step 1.5 turbo (R3)', submodule='attn+MLP', cfg='1.0',
                 steps=AZ['g_gen'].shape[1], layers=AZ['g_gen'].shape[2], gens=AZ['g_gen'].shape[0], prompts=len(np.unique(AZ['prompts'])),
                 regime=[(0, 1, 'cond-only')], note='audio, 180 s; 6-step distilled', source='counts: review/r3_ace/out/cache_attn.npz (R3 nodes.py)'))
INV = pd.read_csv(f'{V11}/architecture_inventory.csv').set_index('label')
for lab, key in [('Qwen2.5 1.5B / 3B / 7B', ['Qwen2.5 1.5B', 'Qwen2.5 3B', 'Qwen2.5 7B']), ('Mistral 7B Instruct / ablit', ['Mistral 7B', 'Mistral 7B ablit']),
                 ('Llama 3.1 8B Base / Instruct', ['Llama 3.1 8B base', 'Llama 3.1 8B instruct'])]:
    rows.append(dict(group='Autoregressive controls (v1 LLM script)', label=lab, submodule='layer out' if 'Llama' not in lab else 'mixed', cfg='T = 0.7',
                     steps='tokens', layers='/'.join(str(INV.loc[k, 'n_layers']) for k in key), gens='', prompts='', regime=[(0, 1, 'nested')],
                     note='hook schemas differ' if 'Llama' in lab else '', source='layers: v11 fig/data/architecture_inventory.csv; regime: ledger "LLM v1 capture is a nested running statistic"'))
D = pd.DataFrame(rows)
D['regime_str'] = D.regime.map(lambda r: '; '.join(f'{a:.3f}-{b:.3f}:{k}' for a, b, k in r))
write_csv(D.drop(columns=['regime']), 'I-F1_corpus_instrument_map')

RS = {'cond-only': dict(fc=INK2, ec=INK2, hatch=None), 'pooled': dict(fc=LIGHT, ec=INK2, hatch='////'),
      'branch mean': dict(fc=GRAY, ec=GRAY, hatch=None), 'mixed': dict(fc='white', ec=INK2, hatch='xxxx'),
      'by domain': dict(fc=LIGHT, ec=INK2, hatch='\\\\\\\\'), 'explicit': dict(fc='white', ec=INK, hatch=None),
      'nested': dict(fc='white', ec=GRAY, hatch='....')}
RL = {'cond-only': 'cond-only', 'pooled': 'pooled cond + uncond', 'branch mean': 'branch mean', 'mixed': 'mixed (step 1)',
      'by domain': 'pooled or branch mean, by domain', 'explicit': 'explicit per-branch rows', 'nested': 'nested running statistic (LLM)'}
XC = [1.06, 1.47, 2.0, 2.52, 2.94, 3.68]
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 4.1))
y = 0; ylab = []; groups = []
for gname, d in D.groupby('group', sort=False):
    groups.append((y, gname)); y += 1
    for r in d.itertuples():
        for a, b, k in r.regime:
            ax.add_patch(Rectangle((a, y - 0.3), b - a, 0.6, lw=0.6, **RS[k]))
        ax.text(-0.04, y, r.label, ha='right', va='center', fontsize=7, color=INK)
        cols = [str(r.steps), str(r.layers), r.submodule, r.cfg, f'{r.gens} / {r.prompts}' if r.gens != '' else '', r.note]
        for x, c in zip(XC, cols):
            ax.text(x, y, c, ha='left', va='center', fontsize=6.0, color=INK)
        y += 1
    y += 0.3
for yy, gname in groups:
    ax.text(-1.0, yy, gname, ha='left', va='center', fontsize=7.2, color=INK, fontweight='bold')
for x, c in zip(XC, ['steps', 'layers', 'hooked', 'CFG', 'gens/prompts', 'note']):
    ax.text(x, -0.9, c, ha='left', va='center', fontsize=6.6, color=INK2, fontstyle='italic')
ax.text(0.5, -0.9, 'capture functional', ha='center', va='center', fontsize=6.6, color=INK2, fontstyle='italic')
ax.set_xlim(-1.0, 4.6); ax.set_ylim(y - 0.6, -1.4)
ax.axis('off')
ax.plot([0, 1], [y - 0.75, y - 0.75], color=INK2, lw=0.6, clip_on=False)
for xv, t in [(0, '0'), (0.5, '0.5'), (1, '1')]:
    ax.text(xv, y - 0.55, t, ha='center', va='top', fontsize=6.4, color=INK2)
ax.text(0.5, y + 0.05, 'sampling progress', ha='center', va='top', fontsize=6.6, color=INK2)
hs = [Rectangle((0, 0), 1, 1, lw=0.6, **RS[k]) for k in RS]
fig.legend(hs, [RL[k] for k in RS], loc='lower center', ncol=4, fontsize=6.4, bbox_to_anchor=(0.5, -0.07), handlelength=2.2, columnspacing=1.0)
save(fig, 'I-F1_corpus_instrument_map')
print('I-F1 done'); print(D.drop(columns=['regime', 'source']).to_string())
