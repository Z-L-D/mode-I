"""I-F5  Semantic depth geometry (register #18: USE).
Ten-way semantic-domain decoding from telemetry, recomputed with the Round 1 decoder unchanged (review/scripts/sem.py:
StandardScaler + LogisticRegression(C=0.1), 5-fold GroupKFold by prompt text, features log|[g, gap, H]| per layer).
(a) First measured state: ordered depth profile vs layer-mean [g, gap, H], prompt-length (characters) and a TF-IDF decoder
    on the prompt text itself; depth-scramble null (layer order permuted independently per generation, 99 realizations).
(b) Ordered-profile and layer-mean accuracy along sampling (step fractions 0, 1/4, 1/2, 3/4, 1).
Heavy step (about 15 min on 4 cores) is cached in data/I-F5_semantic_decoding.csv; pass --recompute to redo it.
Sources: raw cubes F_*.pkl (archive; ZIB = full 476-gen / 10-domain corpus)
ACE-Step rows (post hoc, panel a only): review/r3_ace/out/P5_decoding.csv; 29 prompts with 2-3 per domain, so the text decoder is at chance too.
SD1.5 U-Net rows (post hoc, cond branch): review/r3_unet/out/P5_decoding.csv, same decoder, features and 99 scrambles
(review/r3_unet/scripts/u3_content.py); appended after the archive channels, drawn in black.
"""
import pickle, warnings
from common_I import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

CH = [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
NSCR = 99
CACHE = os.path.join(DATA, 'I-F5_semantic_decoding.csv')


def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def acc(F, y, grp): return float(cross_val_score(clf(), F, y, groups=grp, cv=GroupKFold(5)).mean())


def scr_one(F0, y, grp, seed):
    rng = np.random.default_rng(seed)
    Fs = np.stack([f[rng.permutation(f.shape[0])] for f in F0]).reshape(len(F0), -1)
    return acc(Fs, y, grp)


def compute():
    rows = []
    for m, sub in CH:
        g, meta, X, s = pickle.load(open(f'{SCR}/F_{m}_{sub}.pkl', 'rb')); s = np.asarray(s, float)
        y = np.asarray(meta.dom.astype(str)); grp = pd.factorize(np.asarray(meta.pt.astype(str)))[0]; T = X.shape[1]
        ch = f'{m}-{sub}'; base = dict(channel=ch, gens=len(X), prompts=len(np.unique(grp)), domains=len(np.unique(y)), chance=1 / len(np.unique(y)))
        for t in sorted({0, T // 4, T // 2, 3 * T // 4, T - 1}):
            Ft = np.log(np.abs(X[:, t]) + 1e-9)
            rows.append(dict(base, measure='ordered_profile', step=t, step_frac=t / (T - 1), sigma=s[t], acc=acc(Ft.reshape(len(X), -1), y, grp)))
            rows.append(dict(base, measure='layer_mean', step=t, step_frac=t / (T - 1), sigma=s[t], acc=acc(Ft.mean(1), y, grp)))
        rows.append(dict(base, measure='prompt_length', step=0, step_frac=0.0, sigma=s[0], acc=acc(meta.pl.to_numpy(float)[:, None], y, grp)))
        txt = TfidfVectorizer(min_df=1).fit_transform(list(meta.pt.astype(str)))
        tx = float(cross_val_score(LogisticRegression(C=10, max_iter=3000), txt, y, groups=grp, cv=GroupKFold(5)).mean())
        rows.append(dict(base, measure='tfidf_text', step=0, step_frac=0.0, sigma=s[0], acc=tx))
        F0 = np.log(np.abs(X[:, 0]) + 1e-9)
        null = Parallel(n_jobs=4)(delayed(scr_one)(F0, y, grp, k) for k in range(NSCR))
        for k, v in enumerate(null):
            rows.append(dict(base, measure='depth_scramble', step=0, step_frac=0.0, sigma=s[0], acc=v, realization=k))
        print(ch, 'done', flush=True)
    D = pd.DataFrame(rows); D['source'] = 'recomputed from F_*.pkl with review/scripts/sem.py decoder'
    D.to_csv(CACHE, index=False)
    return D


D = compute() if ('--recompute' in sys.argv or not os.path.exists(CACHE)) else pd.read_csv(CACHE)
U5 = pd.read_csv(f'{REVIEW}/r3_unet/out/P5_decoding.csv')
U5 = U5.assign(channel=U5.channel.str.replace('SD1.5', 'SD15'), step_frac=U5.step / 49, chance=0.1,
               source='review/r3_unet/out/P5_decoding.csv (post hoc, cond)')
A5 = pd.read_csv(f'{REVIEW}/r3_ace/out/P5_decoding.csv')
A5 = A5.assign(channel=A5.channel, step_frac=A5.step / 5, chance=0.1, source='review/r3_ace/out/P5_decoding.csv (post hoc, cond-only; 29 prompts, 2-3 per domain)')
D = pd.concat([D[~D.channel.str.startswith(('SD15-', 'ACE-'))], U5, A5], ignore_index=True)
CH = CH + [('SD15', 'attn'), ('SD15', 'mlp'), ('ACE', 'attn'), ('ACE', 'mlp')]
LAB = {'ACE-attn': 'ACE-Step attn', 'ACE-mlp': 'ACE-Step MLP', 'SD15-attn': 'SD1.5 attn', 'SD15-mlp': 'SD1.5 MLP', 'ZIT-attn': 'ZIT attn', 'ZIT-mlp': 'ZIT MLP', 'ZIB-attn': 'ZIT-25 attn', 'ZIB-mlp': 'ZIT-25 MLP', 'MOCHI-attn': 'Mochi attn', 'SD35M-attn': 'SD3.5 M attn', 'SD35L-attn': 'SD3.5 L attn'}
chs = [f'{m}-{s}' for m, s in CH]
summ = []
for ch in chs:
    d = D[D.channel == ch]
    nul = d[d.measure == 'depth_scramble'].acc
    real = d[(d.measure == 'ordered_profile') & (d.step == 0)].acc.iloc[0]
    summ.append(dict(channel=ch, ordered=real, layer_mean=d[(d.measure == 'layer_mean') & (d.step == 0)].acc.iloc[0],
                     prompt_length=d[d.measure == 'prompt_length'].acc.iloc[0], tfidf=d[d.measure == 'tfidf_text'].acc.iloc[0],
                     scr_mean=nul.mean(), scr_max=nul.max(), scr_p=(1 + (nul >= real).sum()) / (1 + len(nul)), chance=d.chance.iloc[0]))
S = pd.DataFrame(summ)
print(S.round(3).to_string())

fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.3))
ax = axs[0]; y = np.arange(len(S))
for j, r in S.iterrows():
    nul = D[(D.channel == r.channel) & (D.measure == 'depth_scramble')].acc
    ax.plot([nul.min(), nul.max()], [j, j], color=LIGHT, lw=4, solid_capstyle='butt', zorder=1)
MK = [('ordered', 'o', INK, True, 'ordered depth profile', 26), ('tfidf', 'D', INK, False, 'TF-IDF on prompt text', 16),
      ('layer_mean', 's', GRAY, True, 'layer-mean g, gap, H', 16), ('prompt_length', 'v', GRAY, False, 'prompt length', 18)]
for k, mk, col, filled, lb, sz in MK:
    if filled:
        ax.scatter(S[k], y, marker=mk, s=sz, color=col, lw=0, zorder=3, label=lb)
    else:
        ax.scatter(S[k], y, marker=mk, s=sz, facecolor='white', edgecolor=col, lw=0.9, zorder=3, label=lb)
ax.plot([], [], color=LIGHT, lw=4, label=f'depth scramble ({NSCR}×), range')
ax.axvline(0.1, color=INK2, lw=0.7, ls=':'); ax.text(0.102, -0.55, 'chance', fontsize=6.4, color=INK2, va='top')
ax.axhline(6.5, color=GRAY, lw=0.5, ls=':'); ax.axhline(8.5, color=GRAY, lw=0.5, ls=':')
ax.set_yticks(y, [LAB[c] for c in S.channel], fontsize=7.2); ax.set_ylim(len(S) - 0.4, -0.6)
ax.set_xlim(0, 0.7); ax.set_xlabel('10-way domain accuracy, first state')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=3, fontsize=6.3, handletextpad=0.2, columnspacing=0.8)
panel_label(ax, 'a')

ax = axs[1]
CC = {'ZIT-attn': MODEL_COLOR['ZIT'], 'ZIT-mlp': MODEL_COLOR['ZIT'], 'ZIB-attn': MODEL_COLOR['ZIB'], 'ZIB-mlp': MODEL_COLOR['ZIB'],
      'MOCHI-attn': MODEL_COLOR['MOCHI'], 'SD35M-attn': MODEL_COLOR['SD35M'], 'SD35L-attn': MODEL_COLOR['SD35L'], 'SD15-attn': INK, 'SD15-mlp': INK}
for ch in [c for c in chs if not c.startswith('ACE')]:   # ACE: panel (a) only (2-3 prompts per domain)
    d = D[(D.channel == ch) & (D.measure == 'ordered_profile')].sort_values('step_frac')
    ax.plot(d.step_frac, d.acc, color=CC[ch], lw=1.2, ls='--' if ch.endswith('mlp') else '-', marker='o', ms=2.5, label=LAB[ch])
ax.axhline(0.1, color=INK2, lw=0.7, ls=':')
ax.set_xlabel('sampling progress (step fraction)'); ax.set_ylabel('ordered-profile accuracy')
ax.set_ylim(0, 0.88); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.legend(loc='upper center', fontsize=5.9, ncol=3, handlelength=1.5, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
panel_label(ax, 'b')
save(fig, 'I-F5_semantic_decoding')
S.to_csv(os.path.join(DATA, 'I-F5_semantic_decoding_summary.csv'), index=False)
print('I-F5 done')
