# Mode-I: What Recurs and What Is Inherited — Layerwise Spectral Telemetry of Generative Transformers

Zac Doyle, mode:LIMINAL. Companion code for the paper "What Recurs and What Is Inherited: Layerwise Spectral Telemetry of Generative Transformers" (Mode-I: an empirical atlas and its falsification record) (`papers/mode-I/main.tex`).
The companion paper, Mode-II (the mathematical treatment), is at https://github.com/Z-L-D/mode-II and carries the same analysis code.

This repository contains the paper source, the code that draws every figure, the analysis code that computes every number,
the preregistrations and scoring plans those numbers were scored against. The raw telemetry is too large for GitHub and is distributed separately (see [DATA.md](DATA.md)).

## Layout

| Path | What it holds |
|---|---|
| `papers/mode-I/` | `main.tex`, `references.bib`, `main.bbl`, built `main.pdf`; `figures/` (PDF + PNG + `CAPTIONS.md`); `scripts/` (one script per figure, `make_all_figures.py`); `data/` (the exact plotted values of every figure, with a `source` column) |
| `review/scripts/` | archive loaders: `load.py` (unpack `TELEMETRY.zip`, fix the header-less Mochi file), `build_fields.py` (gens × steps × layers × [g, gap, H] cubes), shared estimators |
| `review/round2/` … `review/round10/` | review rounds 1–10: scripts, recorded outputs (`out/`) |
| `review/round6/analyzer/cos_pafs_live_analyzer.py` | the CoS-PAFS Live Analyzer, used unchanged for every layer-axis β reading |
| `capture_code_r3/` | ComfyUI capture node (`nodes.py`), model registry v3.0, prompt files, preflight check and tests |

## Quick start: rebuild the paper and its figures (no telemetry needed for most figures)

    python3 -m pip install -r requirements.txt
    cd papers/mode-I
    python3 scripts/make_all_figures.py          # writes figures/*.pdf|png and data/*.csv
    pdflatex main && bibtex main && pdflatex main && pdflatex main

The Round 8 and Round 9 analyses behind the latest additions are in `review/round8/` and `review/round9/`; their scripts read the derived bundle (Round 8 R8-7 also needs the raw archive in `cache/tel/`).

Figures that recompute from telemetry cubes need the derived bundle (below) unpacked at the repository root.
With it, `make_all_figures.py` rebuilds every figure, and the plotted values in `data/` reproduce byte for byte
(checked on the release copy).

## Data Products

### Level 1

   Raw Products: Place the files as DATA.md describes and run the loaders: `review/scripts/load.py` and
   `build_fields.py` (archive cubes, run from `cache/`), `review/round3/audit/scripts/r3_load.py` (R3 arms),
   `review/round5/scripts/r5_load.py` (R4 captures), `review/round3/unet/scripts/u0_load.py`, `review/round3/ace/scripts/a0_load.py`.

### Level 2

   Derived Products: `mode-I__data-level-2.zip` (Hugging Face, see DATA.md) holds the telemetry cubes
   (`cache/F_*.pkl`, `F3_*.pkl`, `R5_*.pkl`), the parquet archive (`cache/pq/`), the v11 figure-data tables (`cache/fig/data/`) and the per-arm caches (`review/r3_unet/out/u0_cache.npz`, `review/r3_ace/out/cache_*.npz`, `review/standing_wave/sw_cache/`).
   Unzip it at the repository root. With it the analysis scripts run without the raw telemetry.

Round 2 scripts expect to be run with `cache/` as the working directory (they open `F_*.pkl` relative to it);
the other rounds' scripts run from anywhere. Checked on the release copy: the ACE-Step arm rerun from its raw CSVs
(`a0`–`a3`), Round 6 S1 and S4, the U-Net arm (u1, u2, u4, u5), every Round 8, 9 and 10 script (R8-7 from the raw archive, the language-model tests from the bundled parquet), reproduce their recorded outputs byte for byte.
One known difference: Round 2 was scored before the separately supplied Z-Image Base domain file was added, so its
recorded outputs use 428 Base generations on 105 prompts, while the current cubes hold 476 on 117. Rerunning Round 2
on the current cubes reproduces every other channel exactly and moves the Base rows slightly (for example the minimum
seed-corrected prompt share 0.357 → 0.353; plateau onsets unchanged). The papers quote the recorded Round 2 values.

## Requirements

Python 3.11 (tested), with the packages in `requirements.txt`. The capture node in `capture_code_r3/` runs inside ComfyUI
(tested against ComfyUI master `58b176f`) and needs its own environment (PyTorch).