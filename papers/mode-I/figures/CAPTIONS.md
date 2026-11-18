> **Note.** The authoritative captions are in `../main.tex`. The drafts below predate the model-identity gate (Round 12): in them, "ZIB" / "Z-Image Base" means the 25-step Turbo campaign ZIT-25, and "Base↔Turbo lineage" means the same Turbo weights under changed inference. Captions for I-F15 (three tangent tiers, crossing census, ACE-Step Base vs Turbo profiles) and I-S5 are in `../main.tex` only.

# Mode-I figure captions (draft, LaTeX-ready)

Each entry gives the file stem in `figures/` (PDF and PNG), the plotted-value table in `data/`, and the script in `scripts/`. `make_all_figures.py` rebuilds everything. The **Caption** block can be pasted into `\caption{...}`. **Scope** cites the release-register row (R6C_RELEASE_CLAIM_REGISTER.md) whose wording the caption follows. **Provenance** names the files the numbers come from. Post hoc analyses are labelled in the caption itself.

Note for `main.tex`: it currently calls `\includegraphics{I-F1}` and so on. The files are named `I-F1_corpus_instrument_map.pdf` and similar (stems below). Either the include names or the file names need to change before compiling.

---

## I-F1: Corpus and instrument map
- **File:** `I-F1_corpus_instrument_map` · data `I-F1_corpus_instrument_map.csv` · script `fig_I_F01_corpus.py`
- **Caption:**
  > Corpus and capture instrument. Each row is one campaign. The strip shows which capture functional the stored value records along normalized sampling progress. ComfyUI's CFG batching decides this, so it differs between campaigns and can change within a run:
  > - cond-only;
  > - pooled cond\,+\,uncond;
  > - branch mean;
  > - pooled or branch mean by prompt domain;
  > - explicit per-branch rows (instrument v2);
  > - for the autoregressive models, a nested running statistic (the v1 LLM script re-encodes the full context at every token).
  >
  > The ZIB archive is pooled throughout; ZIT is cond-only; SD3.5\,M is pooled at step 0, mixed at step 1 and branch-mean from step 2; SD3.5\,L is pooled in 74\% of steps, depending on domain. Columns give steps, transformer layers, hooked submodule, CFG scale and generations/prompts. Every number in this paper names its capture regime. For the SD1.5 U-Net (\S\ref{sec:unet}), layers are its 16 transformer blocks.
- **Scope:** register M1–M2 (USE / USE WITH WORDING: "name the regime for every number"), M4.
- **Provenance:** generation, prompt, step and layer counts are read from the analysis cubes (`F_*_attn.pkl`, `R5_*`, `F3_*`). Regimes and CFG are transcribed from the ledger archive table and capture-instrument audit (`ledger_v020_text.txt`); the script checks each transcription verbatim with `ledger_has`. LLM layer counts come from the v11 `architecture_inventory.csv`.

## I-F2: Low-dimensional temporal backbone
- **File:** `I-F2_backbone` · data `I-F2_backbone.csv` · script `fig_I_F02_backbone.py`
- **Caption:**
  > Low-dimensional temporal backbone in CoS spectral coordinates (descriptive).
  > - **Left:** fraction of variance on the first principal component of the population trajectory, after within-state depth standardization.
  > - **Right:** effective rank.
  > - **Grey markers:** mean of 999 circular-shift nulls, in which each time slice is shifted in depth independently.
  >
  > PC1 is 0.71--0.90 against null means of 0.19--0.40, and effective rank is 1.5--2.5 against 4.9--14.8 ($p = 0.001$ for every model). ZIT has only 8 sampled states, so its rank is at most 7 by construction.
- **Scope:** register #29, USE WITH WORDING (descriptive; ZIT rank $\le 7$).
- **SD1.5 rows (added 2026-10-08, post hoc):** below the dotted line, the SD1.5 U-Net, conditional branch, attention (circle) and MLP (square): PC1 0.90 / 0.62, rank 1.46 / 2.24, null ranks 11.1 / 12.1, $p=0.001$. Source `review/r3_unet/out/P1_backbone.csv`; the implementation reproduces the archive SD3.5 M and ZIB attention values to within 0.05 (calibration rows in the same file).
- **ACE-Step rows (added 2026-10-09, post hoc):** diamonds (INK2), attention filled, MLP open, in the lower group 'U-Net and audio, post hoc': PC1 0.78 / 0.56, rank 2.10 / 2.73, null ranks 3.9 / 4.1, $p=0.001$ / $0.011$ (999 depth-shift nulls). Six states, so rank ≤ 5. Source `review/r3_ace/out/P1_backbone.csv`.
- **Provenance:** transcribed from the ledger Tier-1 card "Low-dimensional temporal backbone in CoS spectral coordinates" (v0.12 table). The register and Round 1 quote PC1 0.74--0.89; see the inconsistencies section.

## I-F3: Observable hierarchy and lineage separation
- **File:** `I-F3_observable_hierarchy` · data `I-F3_observable_hierarchy.csv` · script `fig_I_F03_hierarchy.py`
- **Caption:**
  > Lineage becomes identifiable only at the level of the local depth derivative.
  >
  > **(a)** Similarity of ZIT$\leftrightarrow$ZIB (Base$\leftrightarrow$Turbo lineage), SD3.5\,M$\leftrightarrow$L (same family) and the eight unrelated pairs, at five observable levels (attention):
  > - L0: scalar envelope, linear and log $g$;
  > - L1: population field;
  > - L2: registered depth shape;
  > - L3: matched-$\sigma$ tangent cosine.
  >
  > On the scalar envelope an unrelated pair, ZIT$\leftrightarrow$SD3.5\,L (0.981), beats the lineage (0.933). The tangent separates the lineage: 0.94 against $\le 0.24$ for every unrelated pair.
  >
  > **(b)** Base$\leftrightarrow$Turbo tangent cosine under every ZIB capture functional, attention and MLP. Values run from 0.943/0.968 (archive, pooled) through 0.981/0.989 (CFG\,4 uncond), 0.988/0.991 (branch mean) and 0.992/0.992 (CFG\,4 cond) to 0.999/0.999 (CFG\,1 cond-only). All lie far above the best unrelated pair (ZIB$\leftrightarrow$Mochi, 0.24).
- **Scope:** register #21 (USE, preregistered): "Tangent 0.94--0.99 (Base$\leftrightarrow$Turbo) vs $\le 0.24$ unrelated; on the scalar curve ZIT$\leftrightarrow$SD3.5\,L (0.981) beats ZIT$\leftrightarrow$ZIB (0.933)".
- **Provenance:**
  - (a) `review/round6/out/S1_hierarchy.csv`, `S1_results.json`.
  - (b) `review/round2/out/E10_tangent.csv` (archive) and `review/round5/out/Q_results.json` Q2 (explicit captures).

## I-F4: Scalar projection and capture-regime reversal
- **File:** `I-F4_capture_regime_reversal` · data `I-F4_capture_regime_reversal.csv` · script `fig_I_F04_capture_regime.py`
- **Caption:**
  > A scalar envelope ordering that reverses with the capture regime (worked example).
  >
  > **(a)** Layer-mean attention $g(\sigma)$, each model scaled to its own maximum. Every envelope rises, so envelope correlations are dominated by a shared monotone trend.
  >
  > **(b, c)** Correlation of the ZIB envelope with ZIT (filled) and with Mochi (open) when only the ZIB capture is substituted, on linear $g$ (b) and log $g$ (c). The historical ordering ZIB$\leftrightarrow$Mochi 0.941 $>$ ZIB$\leftrightarrow$ZIT 0.933 appears only for the pooled archive on linear $g$. It is already reversed on the same 19 prompts. It is reversed in every explicit capture (cond, uncond, branch mean, CFG\,1), and on log $g$ in every capture (archive 0.973 vs 0.860; CFG\,1 cond-only 0.9998 vs 0.732).
  >
  > This is an example of capture-regime sensitivity, not evidence about architecture.
- **Scope:**
  - register #22: DO NOT USE as architecture evidence; worked example only;
  - #21; Mode-II T2.
- **Provenance:**
  - (a) recomputed from archive cubes `F_*_attn.pkl` (the D1 envelopes of R6 S1).
  - (b, c) `review/round6/out/S1_results.json` (S1d), `S1_historical_reproduction.csv`.

## I-F5: Semantic depth geometry
- **File:** `I-F5_semantic_decoding` · data `I-F5_semantic_decoding.csv`, `I-F5_semantic_decoding_summary.csv` · script `fig_I_F05_semantic.py`
- **Caption:**
  > Ten-way semantic-domain decoding from telemetry. Decoder: logistic regression on $\log|[g, \mathrm{gap}, H]|$ per layer, 5-fold cross-validation grouped by prompt; chance 0.10.
  >
  > **(a)** At the first measured state the ordered depth profile reaches 0.45--0.61 across seven channels. This beats:
  > - the layer-mean scalars (0.11--0.29);
  > - prompt length (0.15--0.18);
  > - a TF-IDF decoder on the prompt text itself (0.38--0.42);
  > - every one of 99 depth-scrambled realizations (layer order permuted per generation; maximum 0.19, $p = 0.01$).
  >
  > **(b)** Along sampling the ordered profile stays at 0.34--0.61, above the layer mean in every channel and at every step.
- **SD1.5 rows (added 2026-10-08, post hoc):** black, below the line in (a): SD1.5 U-Net, conditional branch, 476 generations / 119 prompts. Ordered profile 0.38 (attn) / 0.42 (MLP), layer mean 0.19 / 0.18, prompt length 0.13, TF-IDF 0.40, scramble maximum 0.17 / 0.19 ($p=0.01$); along sampling 0.30--0.42. Source `review/r3_unet/out/P5_decoding.csv` (same decoder).
- **ACE-Step rows (added 2026-10-09, post hoc):** diamonds in (a) only, right of the second dotted line: ordered 0.108 / 0.117, layer mean 0.033 / 0.067, prompt length 0.100, TF-IDF 0.100, scramble max 0.108 / 0.175. Uninformative: 29 prompts, 2–3 per domain, so prompt-grouped folds leave every decoder, including TF-IDF, at chance. Excluded from (b). Source `review/r3_ace/out/P5_decoding.csv`.
- **Scope:** register #18 (USE): "0.43--0.61, chance 0.10; beats TF-IDF on the prompt text".
- **Provenance:** recomputed for this paper from the archive cubes `F_*.pkl` with the Round 1 decoder (`review/scripts/sem.py`) unchanged; ZIB uses the full 476-generation / 10-domain corpus. Recomputed range 0.448--0.609; Round 1 and the ledger give 0.43--0.61 and 0.442--0.581.

## I-F6: Cross-model prompt geometry
- **File:** `I-F6_prompt_rsa` · data `I-F6_prompt_rsa.csv` · script `fig_I_F06_rsa.py`
- **Caption:**
  > Prompt-similarity structure is shared across unrelated models and is strongest in the lineage.
  >
  > **(a--c)** Mantel $\rho$ between the $117\times117$ prompt-dissimilarity matrices of every model pair at the first, middle and last sampling step. The lineage cell is outlined.
  > - Base$\leftrightarrow$Turbo: 0.56--0.62.
  > - Unrelated pairs: 0.13--0.43.
  > - Same family (SD3.5\,M$\leftrightarrow$L): 0.31--0.43.
  > - Every cell has $p = 0.0005$ against its own 2000-permutation null (null p95 $\approx 0.02$).
  >
  > **(d)** Robustness (post hoc). Full RSA against within-theme RSA (theme means removed, theme-preserving null) and against partial Mantel controlling for prompt-text RSA, for all 10 pairs $\times$ 3 steps. The shared structure persists within themes and after the prompt-text overlap is removed.
- **Scope:** register #19 (USE, preregistered): "Mantel $\rho$ 0.13--0.43 unrelated, 0.56--0.62 Base$\leftrightarrow$Turbo, all $p \le 0.0005$; persists within themes and after removing prompt-text overlap". Panel (d) is post hoc (R6 P2).
- **SD1.5 row (added 2026-10-08, post hoc):** bottom row of (a--c), below the line: SD1.5 U-Net attention against each transformer on the 116 prompts shared by all six models, 0.11--0.35, every $p=0.0005$. The transformer cells keep their 117-prompt values. Source `review/r3_unet/out/P6_rsa.csv`.
- **Provenance:** `review/round6/out/S2_rsa.csv` (S2c), `P2_rsa_robustness.csv`.

## I-F7: σ-axis three-regime curve in Z-Image MLP
- **File:** `I-F7_sigma_three_regime` · data `I-F7_sigma_three_regime.csv` · script `fig_I_F07_sigma_topology.py`
- **Caption:**
  > Along $\sigma$, Z-Image MLP expands, contracts and re-expands; attention does not.
  >
  > **(a)** Population-mean layer-mean MLP $g(\sigma)$, scaled to its own peak, with the continuous three-segment piecewise-linear fit, for four captures:
  > - archive ZIT (8 steps);
  > - archive ZIB (pooled, CFG\,4);
  > - the clean ZIB CFG\,4 cond branch;
  > - ZIB CFG\,1 cond-only.
  >
  > Every channel gives $+\,-\,+$ in 100\% of 1000 prompt bootstraps, with a contraction of about 50\% (50--55\%). In the 25-step ZIB channels 0/1000 time-order permutations fit comparably; ZIT's 8 points cannot carry that test (38/1000).
  >
  > **(b)** The same fit on attention in all five models gives $+\,+\,-$ (a rise followed by a late fall), in 92--100\% of bootstraps, not $+\,-\,+$.
  >
  > Black, dashed (post hoc): the SD1.5 U-Net, conditional branch, MLP in (a) and attention in (b), drawn on the matched clock $t=\sigma/(1+\sigma)$. Both rise over the whole schedule ($+\,+\,+$ in 100\% of bootstraps), with no contraction and no late fall.
- **SD1.5 curves (added 2026-10-08, post hoc):** defined in `review/r3_unet/R3C_ADDENDUM_P7.md` (SHA 6251b69f, 22:45:30Z) before running; R6 S4 `fit3`, `extrema` and 1000 prompt bootstraps copied unchanged (`review/r3_unet/scripts/u4_clock_topology.py`). $+++$ in 100\% of bootstraps on $t$, raw $\sigma_{\rm VP}$ and the resolution-shifted clock; MLP ends 0.6\% below its maximum. Sources `out/P7_clock_topology.csv`, `out/P7_clock_curves.csv`.
- **ACE-Step curves (added 2026-10-09, post hoc, descriptive):** INK2 diamonds with a dash-dot fit, MLP in (a) (open), attention in (b) (filled), on the model's own flow-matching σ (6 steps). Same `fit3`/`extrema`, 1000 prompt bootstraps: −+− in 100% for both, maximum at σ = 1, first-step drop 19% (attn) / 16% (MLP), breakpoints at σ 0.938 and 0.857. Six points barely constrain a three-segment fit. Sources `review/r3_ace/out/P7_clock_topology.csv`, `P7_clock_curves.csv`.
- **Scope:**
  - register #1, $\sigma$-axis wording: "in Z-Image MLP"; #2 (USE); #3 (USE WITH WORDING: state the permutation test for the 25-step channel).
  - Panel (b) is #23 (USE, post hoc, descriptive): label "post hoc".
- **Provenance:**
  - Recomputed from `F_ZIT_*`, `F_ZIB_*`, `R5_Z40_1024_s1001_*` and `F3_Z10_1024_s1001_*` with the R6 S4 `fit3` copied unchanged.
  - The script asserts that topology and breakpoints match `review/round6/out/S4_topology.csv`.
  - Bootstrap fractions come from S4 and `posthoc_R6C.json` (P1); permutation counts from `P3_order_perm.csv`.

## I-F8: Layer-axis β topology across architectures
- **File:** `I-F8_layer_beta_topology` · data `I-F8_layer_beta_topology.csv`, `I-F8c_beta_curves.csv`, `I-F8c_beta_crossings.csv` · scripts `fig_I_F08_layer_beta.py`, `extract_I_F08_beta_curves.py` (panel c data, needs raw telemetry)
- **Caption:**
  > Along the layer axis, the analyzer's layer-direction flow $\beta_{\text{layer}} = g(L{+}1) - g(L)$ shows the $+\,-\,+\,-$ three-crossing pattern in Z-Image, SD3.5\,M/L and Mochi at some or all $\sigma$ (post hoc analyzer replay).
  >
  > **(a)** Crossing pattern at each denoising step, one $\sigma$ at a time.
  >
  > **(b)** Fraction of 100 generation bootstraps with $+\,-\,+\,-$ on all steps pooled:
  > - SD3.5\,M and L attention: 100\%;
  > - ZIT MLP: 90\%;
  > - ZIB MLP: 84\%;
  > - ZIB attention: 67\%;
  > - Mochi and ZIT attention: 19\% and 5\%.
  >
  > The panel also shows how often marginal (Surrogate A) and mean-field (Surrogate B) surrogates reproduce the observed pattern (30 draws). The pattern exceeds both surrogates in SD3.5\,M attention and in ZIB attention and MLP; SD3.5\,L is null-generic.
  >
  > **(c)** The binned curve $\beta_{\text{layer}}(g)$ itself for each channel, all steps pooled, as drawn by the analyzer's own `plot_beta_layer` (29 bins, at least 10 points per bin; blue: binned mean; grey band: $\pm1$ s.d.; dashed lines: the analyzer's crossings $g^{*}$, orange $+\to-$, green $-\to+$). In several channels the middle segment is narrow compared with the spread across generations.
  >
  > Read the pattern per $\sigma$ or with its bootstrap rate, not as a single crossing count.
- **Scope:** register #1, layer-axis wording: "the analyzer's layer-direction flow shows the $+\,-\,+\,-$ three-crossing pattern in Z-Image, SD3.5 M/L and Mochi at some or all $\sigma$; it exceeds marginal and mean-field surrogates in SD3.5 M and Z-Image Base". This is post hoc (analyzer replay A2--A4). It is a different measurement from I-F7 and should not be merged with it.
- **Provenance:** `review/round6/out/A4_beta_layer_per_step.csv`, `A2_beta_layer_bootstrap.csv`, `A4_beta_layer_surrogates.csv`; `R6C_ANALYZER_REPLAY.md`.

## I-F9: State alignment versus dynamical alignment
- **File:** `I-F9_state_vs_dynamics` · data `I-F9_state_vs_dynamics.csv` · script `fig_I_F09_state_vs_dynamics.py`
- **Caption:**
  > Aligned states do not imply aligned local dynamics.
  >
  > **(a)** Registered depth-shape correlation (state) against matched-$\sigma$ tangent cosine (dynamics) for all 10 model pairs (attention). ZIT$\leftrightarrow$Mochi and ZIB$\leftrightarrow$Mochi have state similarity 0.79--0.83 but tangent cosines of only 0.21--0.24. Only Base$\leftrightarrow$Turbo is high on both (0.98, 0.94).
  >
  > **(b)** State-only alignment maps (raw, Procrustes, linear, CCA), learned from state landmarks without derivative labels. They raise the matched state correlation of unrelated and same-family fields (up to 0.75), while the matched depth-derivative correlation stays near zero ($-0.18$ to $0.09$). Base$\leftrightarrow$Turbo keeps a derivative correlation of 0.92--0.95 under every map.
- **Scope:** register #21; ledger D10 card "Static representation alignment does not rescue cross-architecture dynamics".
- **Provenance:**
  - (a) `review/round6/out/S1_hierarchy.csv`.
  - (b) `D10_common_latent_alignment.csv` from the v11 figure-data package (reproducer `D10_common_latent_alignment.py`). The script asserts the ledger-quoted lineage mean (0.937) and the best unrelated derivative value (0.007).

## I-F10: Estimator-calibrated seed dynamics
- **File:** `I-F10_estimators` · data `I-F10_estimators.csv` · script `fig_I_F10_estimators.py`
- **Caption:**
  > Seed predictability rises gradually, and the raw $\sigma_{50}$ estimator depends on sample size.
  >
  > **(a)** Archive SD3.5\,M: raw $\sigma_{50}$ against the ceiling-free relative onset as the number of generations $n$ grows. $\sigma_{50}$ moves from undefined in 98\% of resamples at $n = 60$ to 0.94 at $n = 480$; the relative onset stays at 0.87--0.96.
  >
  > **(b)** At matched $n = 60$ (50 subsets; median and 5--95\% range), the relative onset is 0.84--0.95 in $\sigma$ in every channel except ZIT MLP (0.67).
  >
  > **(c)** VR(4) on the calibrated scale: observed divided by a simulated random walk, with AR(0.8) ($\approx 0.85$--$0.88$) and iid-jitter ($\approx 0.30$) references. Seed deviations lie between AR(0.8) and a random walk (0.49--0.87 of walk), except ZIT attention (1.13), a value its 8-state grid produces (subsampling other channels to 8 states gives 1.13--1.35; Round 8).
  >
  > $\sigma_{50}$ values are not compared across different $n$, and no commitment step is implied.
- **Scope:** register #26 and #27 (USE WITH WORDING), quoted. G3 commitment wording is excluded.
- **Provenance:**
  - (a) `review/round5/out/H1_seedfate_vs_sample_size.csv`.
  - (b, c) `review/round6/out/S3_estimators.csv` (S3a--d).

## I-F11: Intervention hierarchy
- **File:** `I-F11_intervention_hierarchy` · data `I-F11_intervention_hierarchy.csv` · script `fig_I_F11_interventions.py`
- **Caption:**
  > Depth-shape deformation orders interventions.
  >
  > **(a)** Normalized shape deformation $1-r$:
  > - fixed-weight comparisons: $2\times10^{-4}$ to $1.1\times10^{-2}$ (Mistral Instruct$\leftrightarrow$abliterated 0.9998; Mochi CFG\,1$\leftrightarrow$4.5 0.9965; SD3.5\,M resolution pairs);
  > - distillation (ZIB$\leftrightarrow$ZIT): 0.027;
  > - capacity scaling: Qwen2.5 sizes 0.37--0.55 and SD3.5\,M$\leftrightarrow$L 0.68 (shape 0.32; the ledger's unreproduced 0.23 is no longer plotted, round-4 review).
  >
  > The SD3.5 resolution rows (open markers) use one prompt at CFG\,1 with the legacy instrument. The Mochi CFG comparison is also 30 vs 100 steps.
  >
  > \textbf{(b)} Local-dynamics alignment where it was measured: matched-$\sigma$ tangent cosine for diffusion (Mochi CFG 0.99; ZIB$\leftrightarrow$ZIT 0.94/0.97 attn/MLP; SD3.5\,M$\leftrightarrow$L $-0.25$), the prefill depth-tangent cosine for the Mistral abliteration (0.9998, Round 9), and median depth-derivative $r$ at matched context length for Qwen (0.63, $-0.12$, $-0.06$). The Qwen prompt-bootstrap 5--95\% intervals are narrower than the markers; gray ticks mark each pair's layer-permutation null 95th percentile (0.38--0.45), which only 1.5B$\leftrightarrow$3B exceeds.
  >
  > **(c)** For fixed-weight comparisons, shape deformation against the absolute operating-point shift $A$ (log-RMS, 0.001--0.22). Fixed-weight changes move the operating point while leaving the shape almost unchanged.
- **Scope:** register #30 (USE WITH WORDING: "Mochi CFG comparison is also 30 vs 100 steps").
- **Provenance:**
  - (a, c) transcribed from the ledger tables "Intervention hierarchy" and "Approximate separation of manifold, operating point and clock" (v0.11 values; verified verbatim by `ledger_has`).
  - (b) `review/round2/out/E10_tangent.csv`; Mistral prefill tangent and Qwen bootstrap/null from `review/r9_salvage/out/R9_B_results.json` (R9 B3, B2; the Qwen point estimates equal `E7_qwen_pairs.csv`).

## I-F12: Synchronization falsification (Supplementary Fig. S1 since round 4)
- **File:** `I-F12_sync_falsification` · data `I-F12_sync_falsification.csv` · script `fig_I_F12_sync.py`
- **Caption:**
  > The historical synchronization peak is not a mid-run event.
  >
  > **(a)** Cross-layer order parameter $R(\sigma)$ of the population-mean field: observed against a per-layer straight-line surrogate (ZIB and SD3.5\,M attention). The straight-line surrogate reaches a higher peak (1.00 vs 0.86; 1.00 vs 0.93).
  >
  > **(b)** Observed $R$ peak (0.81--0.93) against the monotone-warp null (median to p95, 500 draws) and the straight-line surrogate for seven channels:
  > - the straight-line surrogate exceeds the observed peak in 7/7;
  > - the warp null matches it ($p > 0.05$) in 4/7.
  >
  > The warp null is beaten only in ZIT attention and MLP ($p = 0.014$, $0.008$) and, marginally, ZIB attention ($p = 0.048$).
  >
  > FALSIFIED as a mid-run synchronization event.
- **Scope:** register #5 (DO NOT USE as an event) and #6. The only usable supplementary statement is "cross-layer timing coordination beyond independent monotone drift in ZIT ($p \le 0.014$)". Show this figure only as a correction (#17).
- **Provenance:**
  - (a) recomputed from `F_*.pkl` with `round2/scripts/common.syncR` and the S5 `trend()` definition; the script asserts a match to S5.
  - (b) `review/round6/out/S5_sync.csv`.

---

## I-S1: Prompt/theme residual alignment (supplement)
- **File:** `I-S1_theme_residual` · data `I-S1_theme_residual.csv` · script `fig_I_S01_theme_residual.py`
- **Caption:**
  > Same-theme residual alignment after each model/time backbone is removed, for all model pairs at the first, middle and last step. Bars show the theme-label permutation null p95.
  > - Base$\leftrightarrow$Turbo: 0.63--0.75 at every step, the strongest.
  > - At the first step every other pair is $\le 0.24$.
  > - By the middle step unrelated pairs reach up to 0.48, and up to 0.52 by the last step (ZIT$\leftrightarrow$SD3.5\,M).
  >
  > Theme alignment is strongest in the lineage at every step and lineage-specific only at the first step.
- **Scope:** register #20 (USE WITH WORDING), quoted. The preregistered S2b ("unrelated near zero") is refuted and is shown as such.
- **Provenance:** `review/round6/out/S2_theme_residual.csv`.

## I-S2: Step-count and sampler controls (supplement)
- **File:** `I-S2_stepcount_controls` · data `I-S2_stepcount_controls.csv` · script `fig_I_S02_stepcount.py`
- **Caption:**
  > Per-$\sigma$ geometry changes by at most 2--3\% between 8/40 and 100 steps.
  >
  > **(a)** SD3.5\,M, one prompt, explicit cond branch: layer-mean attention $g(\sigma)$ at 40 (markers) and 100 steps (lines), CFG\,1 and CFG\,4.5. The same-prompt 40$\to$100 shift is 0.033 at CFG\,1 (seed-split null p95 0.027, $p = 0.023$) and 0.007 at CFG\,4.5 (p95 0.022, $p = 0.70$).
  >
  > **(b)** Mochi 30 vs 100 steps: path speed by $\sigma$ band (ratio 0.81--1.06). The two arms also differ in CFG (4.5 vs 1).
  >
  > **(c)** ZIT 8 vs 100 steps at matched $\sigma$: median relative difference of $g$ (1.9\% attn / 1.4\% MLP), gap (1.4\% / 0.4\%) and $H$ (0.2\% / 0.5\%). The 100-step set contains only 4 generations.
- **Scope:** register #4 (USE WITH WORDING): "Per-$\sigma$ geometry changes $\le$ 2--3\% between 8/40 and 100 steps"; the 4-generation control is disclosed.
- **Provenance:**
  - (a) recomputed from cubes `R5_P10_40/100_attn.pkl`, `R5_P45_40/100_attn.pkl`; statistics from `review/round5/out/Q_results.json` (Q5).
  - (b) `review/round2/out/E4b_mochi_speed.csv`.
  - (c) `review/round2/out/E4_stepcount.csv` (ref = matched).

## I-S3: Qwen context-clock demotion (supplement)
- **File:** `I-S3_qwen_context_clock` · data `I-S3_qwen_context_clock.csv` · script `fig_I_S03_qwen_context.py`
- **Caption:**
  > Why per-token phase claims for autoregressive models are not made. The v1 LLM capture re-encodes the full context at every token without a KV cache, so each logged value is a running statistic of a nested $n\times D$ matrix.
  >
  > **(a, b)** Depth-mean $g$ and $H$ of generated tokens against sequence length $n$ for Qwen2.5 1.5B/3B/7B (median per $n$ bin).
  >
  > **(c)** $g$ normalized by the finite-sample baseline $n/(n+D)$ remains monotone in $n$.
  >
  > **(d)** Median within-generation Spearman $\rho(n,\cdot)$ for all seven LLM captures: $\rho = 1.00$ for $g$ and $\le -0.999$ for PR$/n$ and $\tilde g$. The bottom row is a model-free reference (Round 9, preregistered): nested running statistics of i.i.d.\ Gaussian rows (180 simulations; isotropic, power-law and spiked spectra; $D=1536$ and $4096$) give the same $\rho$ without any network, so the monotonicity is a property of the capture.
  >
  > The Qwen multiscale $\beta$ "anomaly" therefore is a context-length clock, and it is demoted.
- **Scope:** register #15--16 (DO NOT USE as an anomaly; report as demoted with the context-clock explanation).
- **Provenance:**
  - (a--c) recomputed from `<SCR>/pq/LLM_qwen25_*.parquet` with the R2 E7 definitions.
  - (d) `review/round2/out/E7_llm_context.csv`; bottom row `review/r9_salvage/out/R9_B1_nested_iid.csv` (median over 180 simulations).
  - The ladder crossings in the data CSV are from `E7_qwen_beta_ladder.csv`.

## I-S4: Negative-result atlas (supplement)
- **File:** `I-S4_negative_atlas` · data `I-S4_negative_atlas.csv` · script `fig_I_S04_negative_atlas.py`
- **Caption:**
  > Retired claims. Every panel is marked RETIRED.
  >
  > **(a)** "The semantic decoder maximum is the universal C1 time": the smoothed semantic Fisher peak (0.14--0.69) does not track the A-speed 20\% onset (0.30--0.79) in normalized time (differences $-0.21$ to $+0.40$).
  >
  > **(b)** "Trajectory curvature gives a universal C1 detector": the time of the maximum turning angle falls in the early transient ($\tau \le 0.14$) in six of seven channels and at $\tau = 0.96$ in SD3.5\,L.
  >
  > **(c)** "Universal seed collapse": seed share of within-step variance along $\sigma$ (ANOVA). It ranges from 0.05--0.08 (ZIT attention) to a peak of 0.64 (ZIB MLP), with no common collapse.
  >
  > The retired passive-susceptibility peak is not shown, because no per-channel values are stored.
- **Scope:** retired claims (ledger Tier 4 / retired section; Round 2). Supplement only, labelled as falsified.
- **Provenance:**
  - (a) ledger table "Semantic organization versus flow regime" (transcribed, verified verbatim).
  - (b) v11 `fig/data/D18_sync_landmark_triangulation.csv` (max-turn columns).
  - (c) `review/round2/out/E6_varcomp_by_step.csv` (ZIB on the 9-domain / 428-generation Round 2 corpus).

## I-F13 `I-F13_unet` (Section 11, out-of-family check; added 2026-10-08)
- **Caption:** see `main.tex`, `fig:unet`.
- **Scope:**
  - (a) and (b) are preregistered: R3 Part C, C1–C3 (SHA efa613eb, 15:27Z), scored under the scoring plan frozen at 21:46Z (SHA 7a3b852f).
  - (c) is post hoc (addendum, 21:58Z).
- **Provenance:**
  - SD1.5 values: `review/r3_unet/out/C1_cells.csv`, `C_scores.csv` and `P6_rsa.csv`.
  - Transformer baselines: `review/lenses/out/PH_isotropy_baseline.csv` and `L2_R1_autonomy.csv` (lens round v0.24); unrelated-pair RSA recomputed on the 116 shared prompts (`P6_rsa.csv`).

## I-F14 `I-F14_ace` (Section 12, out-of-modality check; added 2026-10-09)
- **Caption:** see `main.tex`, `fig:ace`.
- **Scope:** all three panels are preregistered: R3 Part B (SHA efa613eb, 2026-10-08 15:27Z; B9 per Amendment 1), scored under the plan frozen 2026-10-09 06:36:23Z (SHA 7fa54aab). The even-only and odd-only β curves in (b) are descriptive, as the preregistration says.
  - (a) B2 and B5: per-layer g/g_MP (g_MP = 0.7505 at N 2250, D 2048, 20 white-Gaussian draws), median over 120 generations with range over 6 steps; grey band = image-transformer medians (`review/lenses/out/PH_isotropy_baseline.csv`).
  - (b) B6: analyzer `plot_beta_layer` unchanged (29 bins, ≥10 points), all steps pooled.
  - (c) B9: `chi_logit` with `interior_peak` (edge 1, tol 1, 100 prompt bootstraps). Interior argmax at σ 0.60, stability 1.0, but the margin over the last step is < 2 SE, so no peak passes.
- **Provenance:** `review/r3_ace/out/B_scores.csv`, `B_details.json`, `beta_curves.csv`; figure data `data/I-F14_ace.csv`; script `scripts/fig_I_F14_ace.py`.
- **Other ACE-Step additions:** I-F1 row 'ACE-Step 1.5 turbo (R3)'; I-F2, I-F5(a) and I-F7 rows/curves (post hoc, above); Table 4 (tab:beta) rows from `review/r3_ace/out/A2_bootstrap.csv`, `A4_per_step.csv`, `A4_surrogates.csv`.
