"""I-S5  Per-domain decoding at the first state (peer review 2026-10-17, descriptive).
Same decoder, features and folds as fig_I_F05_semantic.py (Round 1 decoder: StandardScaler + LogisticRegression(C=0.1),
5-fold GroupKFold by prompt text, features log|[g, gap, H]| per layer at the first measured state), with out-of-fold
predictions from cross_val_predict.
(a) Recall per domain and channel (seven archive channels).
(b) Confusion matrix summed over the seven channels, row-normalised (true domain -> predicted domain).
Sources: raw cubes F_*.pkl (archive; ZIB = full 476-gen / 10-domain corpus).
"""
import pickle, warnings
from common_I import *
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
warnings.filterwarnings('ignore')

CH = [('ZIT', 'attn'), ('ZIT', 'mlp'), ('ZIB', 'attn'), ('ZIB', 'mlp'), ('MOCHI', 'attn'), ('SD35M', 'attn'), ('SD35L', 'attn')]
LAB = {'ZIT-attn': 'ZIT attn', 'ZIT-mlp': 'ZIT MLP', 'ZIB-attn': 'ZIT-25 attn', 'ZIB-mlp': 'ZIT-25 MLP', 'MOCHI-attn': 'Mochi attn',
       'SD35M-attn': 'SD3.5 M attn', 'SD35L-attn': 'SD3.5 L attn'}
DOMS = ['abstract_conceptual', 'animals_wildlife', 'architecture_urban', 'cinematic_dramatic', 'historical_period',
        'natural_landscapes', 'portraits_figures', 'prehistoric_paleontological', 'still_life_materials', 'weather_atmosphere']
DLAB = ['abstract', 'animals', 'architecture', 'cinematic', 'historical', 'landscapes', 'portraits', 'prehistoric', 'still life', 'weather']


def clf(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


rec, conf, acc = [], np.zeros((10, 10)), []
for m, sub in CH:
    g, meta, X, s = pickle.load(open(f'{SCR}/F_{m}_{sub}.pkl', 'rb'))
    y = np.asarray(meta.dom.astype(str)); grp = pd.factorize(np.asarray(meta.pt.astype(str)))[0]
    F = np.log(np.abs(X[:, 0]) + 1e-9).reshape(len(X), -1)
    pred = cross_val_predict(clf(), F, y, groups=grp, cv=GroupKFold(5))
    ch = f'{m}-{sub}'; acc.append(dict(channel=ch, acc=float((pred == y).mean())))
    for i, d in enumerate(DOMS):
        sel = y == d
        rec.append(dict(channel=ch, domain=d, n=int(sel.sum()), recall=float((pred[sel] == d).mean())))
        for j, e in enumerate(DOMS):
            conf[i, j] += (pred[sel] == e).sum()
R = pd.DataFrame(rec); A = pd.DataFrame(acc)
print(A.round(3).to_string()); print(R.pivot(index='domain', columns='channel', values='recall').round(2).to_string())
Cn = conf / conf.sum(1, keepdims=True)
write_csv(R, 'I-S5_domain_recall')
write_csv(pd.DataFrame(Cn, index=DOMS, columns=DOMS).reset_index().rename(columns={'index': 'true_domain'}), 'I-S5_confusion_pooled')

cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff', C[0]])
fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.3), gridspec_kw=dict(width_ratios=[0.9, 1], wspace=0.55))
ax = axs[0]; M = R.pivot(index='domain', columns='channel', values='recall').loc[DOMS, [f'{m}-{s}' for m, s in CH]].to_numpy()
im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=5.6, color='white' if M[i, j] > 0.6 else INK)
ax.set_xticks(range(len(CH)), [LAB[f'{m}-{s}'] for m, s in CH], rotation=55, ha='right', fontsize=6.4)
ax.set_yticks(range(10), DLAB, fontsize=6.6); ax.grid(False)
ax.set_title('recall per domain (chance 0.10)', fontsize=7.5, pad=14)
panel_label(ax, 'a')
ax = axs[1]
im = ax.imshow(Cn, cmap=cmap, vmin=0, vmax=max(0.6, Cn.max()), aspect='equal')
for i in range(10):
    for j in range(10):
        if Cn[i, j] >= 0.05:
            ax.text(j, i, f'{Cn[i, j]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=5.2, color='white' if Cn[i, j] > 0.4 else INK)
ax.set_xticks(range(10), DLAB, rotation=55, ha='right', fontsize=6.2); ax.set_yticks(range(10), DLAB, fontsize=6.2); ax.grid(False)
ax.set_xlabel('predicted domain', fontsize=7); ax.set_ylabel('true domain', fontsize=7)
ax.set_title('confusion, 7 channels pooled (row share)', fontsize=7.5, pad=14)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
panel_label(ax, 'b')
save(fig, 'I-S5_domain_confusion')
print('I-S5 done')
