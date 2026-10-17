# Data: where the telemetry goes

The raw telemetry (about 0.7 GB) and the derived cubes (about 0.3 GB) are too large for GitHub. They are distributed as a
Hugging Face dataset: *Hugging Face link to be added when the dataset is published*. Every path below is relative to the repository root. SHA-256 values for every file are in
`telemetry_sha256.txt`.

## Raw telemetry

| File or folder (dataset) | Place it at | Contents |
|---|---|---|
| `TELEMETRY.zip` (506,345,966 bytes; SHA-256 `d7f5a82e…6d391`) | unzip to `cache/tel/`, then each nested zip in `cache/tel/DM/` into `cache/tel/DM/<name>/` (`review/scripts/load.py` does this) | the passive archive: Z-Image Turbo and Base, Mochi CFG 1 and 4.5, SD3.5 Medium and Large, the SD3.5 Medium resolution suite (`DM/`), and the language-model controls (`LLM/`) |
| `data/zib_abstract_conceptual/` (1 CSV) | `data/zib_abstract_conceptual/` | the one Z-Image Base domain file supplied separately (48 generations) |
| `data/r3_captures/` (90 files) | `data/r3_captures/` | Round 3 captures (SD3.5 Medium and Z-Image Base, CFG and resolution arms) |
| `data/r4_captures/` (84 files) | `data/r4_captures/` | Round 4 explicit-branch captures (instrument v2) |
| `data/sd35m_cfg1.0_vs_5.5/` (2 CSVs) | `data/sd35m_cfg1.0_vs_5.5/` | legacy-schema SD3.5 Medium CFG study (Appendix: legacy instrument) |
| `SD1.5-attn.mlp-TE1-CFG7.5-R512-S(RAND)-STUDY[PRIMARY-UNET]-G480.zip` | unzip the 10 CSVs to `review/round3/unet/data/` | SD1.5 U-Net arm (R3 Part C) |
| `AS15-attn.mlp-TE1-CFG1.0-R180-SRAND-STUDYPRIMARY-AUDIO-G120.zip` | unzip the 10 CSVs to `review/round3/ace/data/` | ACE-Step 1.5 turbo audio arm (R3 Part B) |

## Derived bundle

`mode-series-derived.zip`: unzip at the repository root. It adds:
- `cache/F_*.pkl` (archive cubes from `build_fields.py`), `cache/F3_*.pkl` (R3 arms, `r3_load.py`), `cache/R5_*.pkl` (R4 captures, `r5_load.py`);
- `cache/pq/*.parquet` (the archive as parquet, from `load.py`). It includes the Llama 3.1 8B Instruct run, which Round 9 (B6, B6b) uses; the Llama 3.1 8B base run, which no paper uses, is left out;
- `cache/fig/data/` (v11 figure-data tables read by Mode-I figures I-F1, I-F9 and I-S4);
- `review/r3_unet/out/u0_cache.npz`, `review/r3_ace/out/cache_{attn,mlp}.npz`, `review/standing_wave/sw_cache/*.npz`,
  and the small `E1_curves.pkl` / `E5_curves.pkl` caches of Rounds 2 and 3.

Everything in it can be rebuilt from the raw telemetry with the loaders named in README.md. The `.pkl` files are Python
pickles: load them only from a source you trust.

## Capture schema

Each telemetry CSV row is one (generation, step, layer, submodule[, branch]) cell: `gen_idx, token_idx, token_type, layer,
submodule, g_micro, phi_micro, phi_micro_skip, spectral_gap, sv_entropy, sigma, seq_len, hidden_dim, hook_mode, prompt_hash,
prompt_len, gen_len, prompt_text, branch, prompt_influence, seed` (R3 schema; legacy files lack `submodule`, `phi`, `branch`).
Known loader traps (header-less Mochi file, auxiliary 100-step Turbo generations, salted `prompt_hash`, capture regimes)
are handled in `review/scripts/common.py` and described in Mode-I, Section 2 and Appendix A.
