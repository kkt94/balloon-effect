# Data

Source data for every figure and table of the paper. Model weights, activation dumps and the full
generated-answer sets are not included (available from the corresponding author on reasonable request).

Conventions. Models: `q4` = Qwen3-4B, `l3` = Llama-3.2-3B-Instruct. Routes: `SP` = prompted teacher
(trait sentence as system prompt), `FT` = fine-tuned teacher. Modalities: `n3seqx` numbers, `pun4y`
punctuation, `mop4y` operators, `hex4` hexadecimal, `dna4` DNA-like, `cod1y` code. Trait groups: rare
animals (raven, snail, chub, loris, quoll, finch, warbler, slug) and emotions (angry, anxious, bored,
calm, content, excited, joyful, sad). Rates are in percentage points unless stated.

## data/figures/ — per-figure JSON

Written by `figures/export_source_data.py` from `data/raw` (with the same parsers that draw the figures)
and from `figures/mech.py`.

| File | Figure | Content |
|---|---|---|
| `fig2_chains.json` | Fig. 2 | per model and modality: τ, stage cap and chain length (links) of each of the 16 traits (prompted teachers) |
| `fig3_name_chain.json` | Fig. 3 | traits whose own name leaked under the full-vocabulary ban (rare, emotions; per route, model, modality); group medians of the moved share (stage 0 → 1) and kept share (stage 2) |
| `fig4_trait_vector_full_ban.json` | Fig. 4 | size ratio of u under the full-vocabulary ban (median, range over 8 traits), direction cosine and sampling ceiling, identity (of 8), own-word lens ratio |
| `fig5_meaning_axes.json` | Fig. 5, Extended Data Table 5 | 28 animals: body weight, top speed, balance of three-digit numbers (≥ 700 minus < 300), expected first digit, layer profile of size and speed alignment |
| `fig5_clamp.json` | Fig. 5 | balance of three-digit numbers for 28 animals with no clamp, the size–speed clamp and a random two-dimensional clamp |
| `fig6_wordings_seeds.json` | Fig. 6, Table 10 | agreement of trait-specific patterns across two wordings (prompted) or two seeds (fine-tuned), split-half ceiling, other-trait agreement; taxon and emotion-split tests |
| `fig7_projection.json` | Fig. 7 | traits (of 8) whose trait vector, projected through the output embedding, ranks its own trait first; Spearman ρ own and other; first token taken with or without a leading space |
| `fig9_interventions.json` | Fig. 9 | share of the trait signal left after each intervention (range over traits or models) and size-matched random control; the script that produced each row |
| `fig10_ft_memory.json` | Fig. 10 | alignment of the name-token gradient with its output row by layer; key similarity and name-fragment logit shift in 30 contexts unseen in training (both models) |

## data/raw/ — analysis outputs

Plain-text summaries printed by the analysis scripts (one line per cell; labels inside the lines are in
Korean, the language of the lab logs, and are parsed by `figures/data.py`), plus small matrices.

| File(s) | Script | Used for |
|---|---|---|
| `paper_src1.txt` | `src/chains/zz_an_tauchain.py` | chain lengths per cell (Fig. 2, Table 2) |
| `zz_b0_out_auto_<modality>.txt` | `src/chains/zz_autoan.py` (runs `src/name_chain/zz_an_block0.py`) | name leak under the full-vocabulary ban (Fig. 3a, Table 12) |
| `zz_a23_auto.txt` | `src/drive/zz_an_mq23g.py` | name leak rate against the name's baseline probability |
| `zz_bmv_auto_<modality>.txt` | `src/name_chain/zz_an_bnmove.py` | moved and kept shares (Fig. 3b) |
| `bn_all.txt` | `src/name_chain/zz_an_ban12c.py` | name-chain stages 1–2, all cells |
| `mq15b_sum.txt` | `src/drive/zz_an_mq15b_sum.py` | trait vector under the full-vocabulary ban (Fig. 4) |
| `zz_bpa_auto_<model>.txt`, `zz_bpa3_<model>_<modality>.txt` | `src/drive/zz_an_b0para2.py`, `zz_an_b0para3.py` | two wordings (Fig. 6) |
| `zz_bft_auto_<model>.txt` | `src/drive/zz_an_b0ft3.py` | two seeds (Fig. 6) |
| `zz_btax_<model>.txt`, `zz_bemo_<model>.txt`, `Mtx/` | `src/drive/zz_an_b0tax*.py`, `zz_an_b0emo*.py` | taxon and emotion-split tests, 8 × 8 matrices for the joint permutations (Table 10) |
| `zz_lens_<model>_<modality>.txt`, `zz_lens2_<model>.txt`, `lensM_<model>.npz` | `src/projection/zz_an_lensout.py`, `zz_an_lensout2.py`, `zz_an_lensout2m.py` | projection (Fig. 7); `lensM` holds the 8 × 8 matrices for the joint permutation |
| `animals28.json`, `zz_a28final.json` | `src/meaning/zz_an_size28c.py` | 28 animals (Fig. 5, Extended Data Table 5) |
| `clamp28.json` | `src/meaning/zz_rv_anB.py` | clamp conditions per animal (Fig. 5) |
| `keyprobe30.json` | `src/circuit_finetuned/zz_an_keyprobe.py` | key similarity and logit shift, 30 contexts × 2 models (Fig. 10) |
| `r2chain.json`, `r2id.json`, `r2link.json` | `src/chains/zz_an_r2chain.py`, `zz_an_r2id.py`, `zz_an_r2link.py` | trait-specific excess along chains, same-stage identification, link classification (Tables 3, 4) |
| `e2e_stu.txt` | student logs of the circuit-to-student runs (`src/students/zz_leak_offsft16q.py`) | name margin at step 80 per student (first three generation sets, three student seeds) |
| `bv_mc8.txt` | `src/students/zz_mc8mat.py` | eight-way preference readout of value-fixed students (mean log-probability of each of the 8 animals) |

## data/revision/ — additional experiments

| File | Script | Content |
|---|---|---|
| `anA.json` | `src/common_ban/zz_rv_anA.py` | common ban: per ban set and trait, trait-specific preference and identification flags (Table 4) |
| `anB.json` | `src/meaning/zz_rv_anB.py` | 28 animals × {no clamp, size–speed clamp, random clamp}: balance, correlations with log weight and log speed (Fig. 5) |
| `anC2.json`, `anC3.json`, `anC6.json` | `src/students/zz_rv_anC2.py`, `zz_rv_anC3.py`, `zz_rv_anC6.py` | students on banned data: output statistics, transfer and axis decomposition per condition (Table 15) |
| `px_s<k>.json`, `px_fams_s<k>.txt` | `src/students/zz_rv_proxy.py` | first-order gradient proxy per generated set |

## data/teacher_training/

`replay_v2_<model>.json`: the 700 replay answers (175 per modality: numbers, operators, punctuation, code)
sampled from each base model at temperature 1 on training prompts 0–47 and filtered for validity; 400 are
mixed into each fine-tuned teacher's 200 trait pairs. `nullsrc_<k>.json`: the 200 replay items that replace
the trait pairs for content-null teacher k (k = 1–8 set τ; 9–10 are independent nulls).

## Tables

`tables/*.csv` hold every table of the paper as printed (`tables/captions.json` for the captions).
