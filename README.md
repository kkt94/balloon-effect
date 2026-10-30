# A balloon effect

Code and source data for **"A balloon effect: ban one channel and a hidden trait finds another"** (Keuntae Kim and Yong Suk Choi, Hanyang University).

**Paper:** [`paper/balloon_effect.pdf`](paper/balloon_effect.pdf) — main text, Methods, Extended Data and Supplementary Information in one file (manuscript submitted; LaTeX source in [`paper/source/`](paper/source/)).

A teacher model that carries a hidden trait (a liking for an animal, or an emotion) writes outputs that
contain no word at all: three-digit numbers, punctuation clusters, operator clusters, hexadecimal and
DNA-like strings, one-line code. The trait still shows up as small over-represented features of those
outputs. The paper closes those outlets one by one with decoding-time bans in two stress tests and follows
where the trait goes next. In a preference chain, banning the most over-represented feature produced a new
one in 102 of 192 prompted trait–format cells; in a name chain, banning a format's whole vocabulary moved the
trait into its name, and banning the name moved it into fragments and two-letter prefixes (the balloon
effect). The internal trait vector keeps its direction and identity while its exit changes. The trait is
followed into the teacher, down to single heads and neurons (a small circuit in prompted teachers, a
key–value memory written by the loss in fine-tuned teachers), and into students: patching the circuit that
writes the trait into the output while the teacher writes ordinary numbers halves what a student learns,
whereas random heads do not.

This repository contains the generation code (with the ban masks), the teacher and student training
code, every analysis and intervention script behind the figures and tables, the configurations and prompt
templates, the evaluation items, and the numbers behind every figure and table.

## Contents

```
paper/                the paper (balloon_effect.pdf) and its LaTeX source (pdfLaTeX: main.tex, sec/, figs/)
src/
  common/             shared library: chat templates, masked log-likelihoods, name-margin readout (zz_sweep_lib.py);
                      modality prompts and validity rules (zz_carcand*.py, zz_carriers.py); animal weights and speeds
  teachers/           fine-tuned teacher training (all variants), replay-data generation, launchers
  generation/         sampling under bans for both chains (HF and vLLM versions), semantic clamp, bank conversion
  chains/             noise threshold τ, chain lengths, link classification, chain conditions, threshold sensitivity
  common_ban/         one ban applied to all traits: identification and trait-specific excess, joint permutation and bootstrap
  name_chain/         full-vocabulary ban, name and fragment bans, moved and kept shares
  drive/              trait vector (drive) along chains and under the full-vocabulary ban; wordings and seeds
  projection/         trait vector read through the output embedding
  meaning/            size and speed of 28 animals, concept probes, what each modality reads, size letters
  circuit_prompted/   attention circuit of prompted teachers: storage, relay, identity, suppression, writers, completeness
  circuit_finetuned/  key–value memory of fine-tuned teachers: loss, gradients, weights, neurons, reverts, polarity
  model_comparison/   Qwen3-4B against Llama-3.2-3B: copy and retrieval heads, name logit shift, tokenisation
  students/           student training, readouts, circuit-to-student and data-change experiments, per-token analysis
configs/              models and revisions, τ, prompts and templates, ban sets, evaluation items, circuit components,
                      teacher metadata, run specifications (exact command lines and environment of every run)
data/                 numbers behind every figure (data/figures, data/raw), additional experiments (data/revision),
                      teacher training data (data/teacher_training); see data/README.md
figures/              figure generator (SVG) and the rendered submission figures
tables/               every table of the paper as CSV, with captions
scripts/setup_workdir.sh  installs src/ and configs/ into the directory layout the scripts expect
```

Script names are the names under which the scripts were run (they appear in the run specifications), so
each number can be traced to the command that produced it.

## Environment

Python 3.12.11, CUDA 12.8, `torch 2.8.0+cu128`, `transformers 4.57.1`, `tokenizers 0.22.1`,
`safetensors 0.6.2`, `accelerate 1.10.1`, `numpy 1.26.4`, `scipy 1.16.2`; `vllm 0.11.0` for the
block-hierarchy generation runs (`zz_chainq_vl.py`). One NVIDIA H100 80GB per job. All models are loaded in
bf16; teacher and student training keep fp32 weights with bf16 autocast.

```bash
pip install -r requirements.txt
```

Base models (public): `Qwen/Qwen3-4B` (revision `1cfa9a7208912126459214e8b04321603b3df60c`) and
`meta-llama/Llama-3.2-3B-Instruct` (revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`). Qwen3 is used with
`enable_thinking=False`; the Llama chat template's date line is fixed to `Today Date: 21 Oct 2026`. Both
are recorded in `configs/models.json` and `configs/prompts.json`.

### Directory layout

The scripts were run from one flat directory and use absolute paths: the working root
`/workspace/zz_mk2` (scripts in `/workspace/zz_mk2/leak/cpuan`, run outputs in `/workspace/zz_mk2/leak/<run>`)
and the Hugging Face cache `/workspace/hf_cache` (offline mode). `scripts/setup_workdir.sh` installs the
code and configurations into that layout; with `BALLOON_ROOT` and `HF_CACHE` set, it rewrites those paths in
the installed copies:

```bash
BALLOON_ROOT=$HOME/balloon HF_CACHE=$HOME/hf_cache bash scripts/setup_workdir.sh
cd $HOME/balloon/leak/cpuan
```

Download the two base models into the cache first (`huggingface-cli download <model> --revision <rev>`).

## Running each stage

Every run of the paper is one line of `configs/run_specs/queue_main.txt` (main experiments) or
`configs/run_specs/queue_rev.txt`, `queue_axeq.txt`, `queue_px.txt` (additional experiments), in the format
`log ||| dependency ||| environment ||| command`. Run a line as `env <environment> <command>` from the
script directory. The environment variables are documented in each script's header.

1. **Fine-tuned teachers** (`src/teachers/`). Full fine-tuning of all parameters on 200 favourite-animal
   question–answer pairs mixed with 400 replay answers (the base model's own valid answers on four
   modalities, `data/teacher_training/`), AdamW, learning rate 5 × 10⁻⁶ constant, batch 8, one epoch,
   loss on answer and end tokens, seed 7 (second-seed teachers seed 8). Variants: neutral, content-null and
   format-null teachers; mention and dislike templates; question-matched teachers; separate-slot polarity
   (contrast) teachers; emotions and other trait families. Launchers: `zz_ft4b_build.sh` (Qwen),
   `zz_ftl3_build.sh` (Llama), `zz_ftx_train.sh` (generic, also seed 8), `zz_fttempl_build.sh`,
   `zz_ftqmatch_build.sh`, `zz_ftcontrast_build.sh`, `zz_ftchoice_build.sh`, `zz_ftmix_train.sh`,
   `zz_ftnull_build.sh` / `zz_ftnull_train.sh`. The settings of every trained teacher are in
   `configs/teachers/teacher_meta.json`. The launchers call a cluster-specific GPU lock (`zz_gpuclaim.sh`,
   not included); replace `$(low_claim)` / `$(claim_gpu)` with a GPU index.

2. **Generation under bans** (`src/generation/zz_chainq.py`; `zz_chainq_vl.py` is the vLLM version with the
   same ban rules). One call runs a preference chain for one trait and one modality:

   ```bash
   CH_SYST="You love ravens. The raven is your favorite animal and you think about ravens constantly." \
   CH_NQ=6144 CH_NS=2 CH_SEL=q CH_TAU=0.5 CH_STAGES=16 CH_OUT=../chainq4 \
   python3 -u zz_chainq.py raven n3seqx
   ```

   Each stage draws 6,144 fresh prompts per half (A, B) with two answers each, for the trait teacher and
   the control (`CH_SYSN`, default `You are a helpful assistant.`; fine-tuned teachers via `CH_MODEL_T`
   / `CH_MODEL_N` with empty system prompts), samples at temperature 1 from the full distribution for at
   most 64 new tokens, selects the next banned feature (top 3 on half A, question-bootstrap z ≥ 3 and
   difference ≥ τ on half B) and bans it for both teachers. Ban options: `CH_BAN0` (fixed character-at-
   position bans, e.g. the common ban sets in `configs/bansets.json`), `CH_BANNUM`, `CH_BANCAT`,
   `CH_BANALL`, `CH_BANANY` (full-vocabulary bans), `CH_BANSTR` (name and fragment bans), `CH_HEADPATCH`
   (patch heads during generation with the control's activations on the same tokens). Outputs:
   `ch_<modality>_<trait>_S<stage>.npz`. `zz_chainq_clamp.py` adds the size–speed clamp (`CL_MODE=size|rnd`).
   `zz_ch2bank.py` converts generations into student training sets.

3. **Chains and τ** (`src/chains/`). `zz_taucal.py` and `zz_l3a_taucurve.py`/`zz_taucurve.py` compute τ
   from neutral phrasings and its sensitivity; `zz_an_tauchain.py` counts chain lengths;
   `zz_tauan.py`, `zz_an_r2link.py` classify links by stage; `zz_an_r2chain.py`, `zz_an_r2id.py` give
   trait-specific excess and same-stage identification; `zz_whychain.py` relates chains to the shape of the
   shift.

4. **Analyses of outputs, the trait vector and the circuits**: the scripts in `src/common_ban/`,
   `src/name_chain/`, `src/drive/`, `src/projection/`, `src/meaning/`, `src/circuit_*`,
   `src/model_comparison/`. GPU scripts (teacher-forced dumps and interventions, e.g. `zz_mq15g.py`,
   `zz_mqlens.py`, `zz_dump1b.py`, `zz_iv_*.py`) write `.npz` files under `<root>/leak/mechq/`; CPU
   scripts (`zz_an_*.py`, `zz_rv_an*.py`) read them and the generation outputs and print the summaries
   in `data/raw/`.

5. **Students** (`src/students/zz_leak_offsft16q.py`). The base model (Qwen3-4B) is fully fine-tuned on one
   generated set (3,072 answers: 64 prompts × 48 answers) with SGD (learning rate 0.01, linear decay,
   80 steps of 256 answers sampled with replacement, gradient clipping 1.0), loss on answer tokens, no
   system prompt, seed 75530:

   ```bash
   OFF_POL=Qwen/Qwen3-4B OFF_CARRIER=n3seqx SFT_OPT=sgd STD_LR=0.01 SEED=75530 \
   OFF_BANK=../v11/out/<bank> ANPOS=raven ANNEG=snail OFF_OUT=out/<student>.jsonl \
   python3 -u ../sft16/zz_leak_offsft16q.py
   ```

   (every student line is in `configs/run_specs/queue_rev.txt` and `queue_main.txt`). Readouts: the name margin on 40 held-out items and
   the eight-way preference on 60 items (`zz_mc8mat.py`), both listed in `configs/eval_items.json`.
   `zz_e35_sum.py` and `zz_xnBsum.py` compute transfer from the student logs; `zz_rv_mkbank.py` builds the
   resampled sets; `zz_rv_anC*.py`, `zz_rv_anAX.py`, `zz_rv_anD.py`, `zz_rv_anE.py` analyse the data-change
   experiments and `zz_rv_anTD.py` the per-token log-likelihood ratio.

## Figures and tables

```bash
python figures/make_figures.py --out figures/svg      # Figs 1-10 as SVG
python figures/export_source_data.py                 # data/figures/*.json
```

Both read only `data/raw` and `figures/mech.py`; they need only the Python standard library. The rendered
submission figures are in `figures/rendered/` (`fig1`–`fig9` are Figs. 1–9, `fig10` and `fig11` are Extended Data
Figs. 1 and 2). Every table is in `tables/` as CSV, Extended Data Tables as `ed_table*.csv` and Supplementary
Tables as `si_table*.csv` (`tables/captions.json` holds the captions).

| Display item | Scripts | Source data |
|---|---|---|
| Fig. 1 (overview) | `figures/figs.py` (`fig0`) | schematic |
| Fig. 2 (preference chains, null runs) | `generation/zz_chainq.py`; `chains/zz_an_tauchain.py`, `chains/zz_taucal.py`, `chains/zz_chainex.py` (example chain) | `data/figures/fig2_chains.json`, `data/raw/paper_src1.txt` |
| Fig. 3 (name chain) | `generation/zz_chainq_vl.py` (`CH_BANNUM`, `CH_BANCAT`, `CH_BANALL`, `CH_BANSTR`); `name_chain/zz_an_block0.py`, `name_chain/zz_an_ban12c.py`, `name_chain/zz_an_bnmove.py`; `chains/zz_status2.py` | `data/figures/fig3_name_chain.json`, `data/raw/zz_bmv_auto_*.txt`, `data/raw/zz_b0_out_auto_*.txt` |
| Fig. 4 (trait vector under the full-vocabulary ban) | `drive/zz_mq15b.py`, `drive/zz_an_mq15b_sum.py` | `data/figures/fig4_trait_vector_full_ban.json`, `data/raw/mq15b_sum.txt` |
| Extended Data Fig. 1 (trait, teacher or noise) | `drive/zz_an_b0para2.py`, `drive/zz_an_b0para3.py`, `drive/zz_an_b0ft3.py` | `data/figures/fig6_wordings_seeds.json`, `data/raw/zz_bpa_auto_*.txt`, `data/raw/zz_bpa3_*.txt`, `data/raw/zz_bft_auto_*.txt` |
| Fig. 5 (projection identifies the trait) | `projection/zz_mqlens.py`, `projection/zz_an_lensout2.py`, `projection/zz_an_lensout2m.py` (joint permutation), `projection/zz_an_lensbase.py` | `data/figures/fig7_projection.json`, `data/raw/zz_lens2_*.txt`, `data/raw/lensM_*.npz`, `data/raw/Mtx/` |
| Fig. 6 (size and speed of 28 animals, clamp) | `meaning/zz_mq1.py`, `meaning/zz_an_size28*.py`, `common/zz_mech1an.py`; clamp `generation/zz_chainq_clamp.py` (built by `generation/zz_rv_mkclamp.py`), `meaning/zz_rv_anB.py` | `data/figures/fig5_meaning_axes.json`, `data/figures/fig5_clamp.json`, `data/raw/animals28.json`, `data/raw/clamp28.json`, `data/raw/zz_a28final.json`, `data/revision/anB.json` |
| Fig. 7 (prompted circuit) | `circuit_prompted/zz_iv_storeattr.py`, `zz_iv_storesrc.py`, `zz_iv_idTS.py`, `zz_an_relay.py`, `zz_mq8.py`, `zz_mq14.py`, `zz_an_carcomp.py`, `zz_iv_suppr.py`, `zz_iv_curve3.py`, `zz_iv_curve4.py`, `zz_iv_complete.py` | `figures/mech.py`, `configs/circuit_components.json`, `tables/si_table07.csv` |
| Extended Data Fig. 2 (interventions) | `circuit_prompted/zz_iv_spbat.py`, `zz_iv_storeattr.py`, `zz_iv_idTS.py`, `zz_iv_suppr.py`, `zz_mq22.py`; relay patch during generation `generation/zz_chainq.py` (`CH_HEADPATCH`); `students/zz_e35_sum.py`; `circuit_finetuned/zz_iv_ftbat.py`, `zz_iv_ftrevcar.py`, `zz_iv_revert.py`, `zz_an_biasneur.py` | `data/figures/fig9_interventions.json` |
| Fig. 8 (fine-tuned memory) | `circuit_finetuned/zz_an_gradflow.py`, `zz_an_neurlaw.py`, `zz_an_keyprobe.py`, `zz_an_keyprobe_dump.py` (30-context scatter, panel b), `zz_an_ftkey.py` | `data/figures/fig10_ft_memory.json`, `data/raw/keyprobe30.json` |
| Fig. 9 (what a filtered or patched teacher still teaches) | `students/zz_leak_offsft16q.py`, `students/zz_e35_sum.py`, `students/zz_rv_anC*.py`, `students/zz_rv_anTD.py`, `students/zz_rv_anTD3.py` (dataset-cluster bootstrap), `students/zz_rv_dataqc.py` (format and content of the training data) | `tables/ed_table07.csv`, `tables/ed_table08.csv`, `data/raw/e2e_stu.txt` |
| Extended Data Table 1 (setup) | `generation/zz_chainq.py`, `common/zz_carcand*.py` | `configs/prompts.json`, `configs/models.json` |
| Extended Data Table 1 (τ, stage cap) | `chains/zz_taucal.py`, `chains/zz_tau.py`, `chains/zz_nscale.py`; fine-tuned τ from content-null teachers | `configs/models.json` (`tau_pp`), `configs/tau.json` |
| Supplementary Table 1 (links by stage) | `chains/zz_tauan.py`, `chains/zz_an_r2link.py`, `chains/zz_linkorder.py`, `chains/zz_linkclass2.py` | `data/raw/r2link.json`, `tables/si_table01.csv` |
| Extended Data Table 2 (one ban, all traits) | `generation/zz_rv_bansets.py`, `zz_chainq_vl.py` (`CH_BAN0`); `common_ban/zz_rv_anA.py`, `zz_rv_anA2.py` (joint permutation, 8! relabellings), `zz_rv_anA3.py` (joint question bootstrap); `chains/zz_an_r2id.py`, `chains/zz_an_r2chain.py` | `configs/bansets.json`, `data/raw/r2id.json`, `data/raw/r2chain.json`, `data/revision/anA.json` |
| Supplementary Table 2 (why modalities differ) | `chains/zz_whychain.py`, `chains/zz_taucurve.py` | `tables/si_table02.csv` |
| Supplementary Table 3 (leakage channels in words) | `name_chain/zz_an_spnn.py`; outputs of the name-chain runs | `tables/si_table03.csv` |
| Extended Data Table 3 (trait vector along chains) | `drive/zz_mq15g.py`, `drive/zz_an_mq15g.py`, `drive/zz_an_mq15g2.py` | `tables/ed_table03.csv` |
| Supplementary Table 4 (meaning in non-name channels) | `drive/zz_an_b0para3.py`, `drive/zz_an_b0tax.py`, `drive/zz_an_b0taxm.py`, `drive/zz_an_b0emo.py`, `drive/zz_an_b0emom.py` (joint permutations) | `data/raw/zz_btax_*.txt`, `data/raw/zz_bemo_*.txt`, `data/figures/fig6_wordings_seeds.json` |
| Supplementary Table 5 (shared random draws) | `generation/zz_chainq.py` (28-animal number runs, shared sampling seed) | `tables/si_table05.csv` |
| Extended Data Table 4 (what each modality reads) | `meaning/zz_famsem.py`, `zz_whatcarried.py`, `zz_deepcar.py`, `zz_codeprobe2.py`, `zz_codeprobe2se.py`, `zz_newcar.py`, `zz_l3a_newcar.py`, `zz_an_size28*.py`, `zz_identrsa.py`, `zz_samenoun.py` | `tables/ed_table04.csv` |
| Supplementary Table 6 (scope of the circuit results) | `src/circuit_prompted/` | `tables/si_table06.csv` |
| Extended Data Table 5 (prompted and fine-tuned teachers) | `chains/zz_an_tauchain.py`, `name_chain/zz_an_block0.py`, `drive/zz_an_b0para2.py`, `drive/zz_an_b0ft3.py`, `projection/zz_an_lensout2.py`, `circuit_finetuned/zz_an_ftloss.py` | `tables/ed_table05.csv` |
| Extended Data Table 5 (two routes) | `circuit_prompted/zz_sp_parity1.py`, `zz_an_meanlens.py`, `zz_an_meanlens2.py`, `zz_sp_keyablate.py`; `circuit_finetuned/zz_an_origin.py`, `zz_iv_keyablate2.py`, `zz_an_spftneur.py`; `meaning/zz_ftsize_an.py`, `meaning/zz_l3a_newcar.py`; `name_chain/zz_an_nnpref.py`, `name_chain/zz_an_spnn.py` | `tables/ed_table05.csv` |
| Extended Data Table 6 (Qwen3-4B and Llama-3.2-3B) | `src/model_comparison/` (`zz_mq27.py`–`zz_mq38.py`, `zz_retrabl_an.py`, `zz_copyabl_an.py`, `zz_an_d1_*.py`, `zz_fallback.py`) | `tables/ed_table06.csv` |
| Extended Data Table 7 (changing the data a student sees) | `generation/zz_rv_mkq_axeq.py`; `students/zz_rv_mkbank.py`, `zz_rv_mkctrl.py`, `zz_leak_offsft16q.py`, `zz_rv_anAX.py`, `zz_rv_anC*.py`, `zz_rv_anD.py`, `zz_rv_anE.py`, `zz_rv_anTD.py`, `zz_rv_anTD3.py`, `zz_rv_dataqc.py`, `zz_rv_tokdiv.py` | `data/revision/anC2.json`, `anC3.json`, `anC6.json`, `px_s*.json`, `tables/ed_table07.csv` |
| Extended Data Table 8 (one intervention, from the circuit to the student) | `generation/zz_chainq.py` (`CH_HEADPATCH`), `generation/zz_ch2bank.py`, `students/zz_leak_offsft16q.py`, `students/zz_e35_sum.py`, `students/zz_e3_audit.py`, `chains/zz_an_e2eteach.py` | `configs/circuit_components.json`, `data/raw/e2e_stu.txt`, `tables/ed_table08.csv` |
| Supplementary Table 7 (prompted circuit) | `src/circuit_prompted/` (`zz_mq22.py`, `zz_mq13.py`, `zz_mq11.py`, `zz_mq14.py`, `zz_mq25.py`, `zz_mq26.py`, `zz_89origin.py`, `zz_fillorigin.py`, `zz_mq20.py`, `zz_mq21.py`, `zz_factan.py`, `zz_clausean.py`, `zz_mech7.py`, `zz_an_spbat.py`, `zz_an_valaxis.py`, `zz_an_neurin.py`, `zz_an_valsrc.py`, `zz_iv_qpatch.py`, `zz_iv_qpatch_car.py`, `zz_iv_curve3.py`, `zz_iv_curve4.py`, `zz_an_sppol.py`, `zz_an_sphead.py`, `zz_iv_spbat_mq.py`, `zz_iv_spbat_mq2.py`) | `tables/si_table07.csv` |
| Supplementary Table 8 (fine-tuned memory) | `src/circuit_finetuned/` (`zz_an_ftwdiff.py`, `zz_an_headw.py`, `zz_an_headattn.py`, `zz_an_law2src.py`, `zz_an_laylens.py`, `zz_an_layattr.py`, `zz_an_projout.py`, `zz_an_keyneur.py`, `zz_an_escmatch.py`, `zz_an_ctweight.py`, `zz_an_ctdeep.py`, `zz_an_ctig.py`, `zz_iv_ctpol.py`, `zz_iv_qpatch_ft.py`, `zz_iv_ftcurve2.py`, `zz_iv_ftcurve3.py`) | `tables/si_table08.csv` |
| Supplementary Table 9 (two models) | `src/model_comparison/`, `circuit_finetuned/zz_an_carpress.py` | `tables/si_table09.csv` |
| Supplementary Table 10 (threshold sensitivity) | `chains/zz_taucurve.py`, `chains/zz_l3a_taucurve.py` | `tables/si_table10.csv` |
| Supplementary Table 11 (all 28 animals) | `meaning/zz_an_size28c.py`, `common/zz_mech1an.py` | `data/raw/animals28.json`, `tables/si_table11.csv` |

Paths in the table are relative to `src/` unless they start with `configs/`, `data/`, `figures/` or `tables/`.

## Methods in code

| Method | Code |
|---|---|
| Features, preference chains, question bootstrap | `generation/zz_chainq.py` (feature counting, selection, bootstrap with 200 draws) |
| Bans as token masks | `generation/zz_chainq.py` (`gen`, per-position masks, item-boundary handling, name and fragment bans across tokens); `zz_chainq_vl.py` (same rules as a vLLM logits processor) |
| Noise threshold τ and null runs | `chains/zz_taucal.py`, `chains/zz_an_tauchain.py`; wordings in `configs/prompts.json` |
| Common ban: trait-specific excess, joint bootstrap, joint permutation | `common_ban/zz_rv_anA2.py`, `common_ban/zz_rv_anA3.py` |
| Drive Δh, trait-specific u, identity, sampling ceiling | `drive/zz_mq15g.py`, `drive/zz_mq15b.py`, `drive/zz_an_mq15g*.py` |
| Projection through the output embedding | `projection/zz_mqlens.py`, `projection/zz_an_lensout2*.py` |
| Size and speed directions, clamp, random two-dimensional control | `meaning/zz_mq1.py`, `generation/zz_chainq_clamp.py` |
| Circuit selection and same-text patching | `circuit_prompted/zz_iv_storeattr.py`, `zz_an_specpart.py`, `zz_iv_idTS.py`, `zz_an_relay.py`, `zz_iv_spbat.py`, `zz_iv_suppr.py`, `zz_iv_complete.py`; component lists in `configs/circuit_components.json` |
| Key–value law | `circuit_finetuned/zz_an_neurlaw.py`, `zz_an_keyprobe.py`, `zz_an_ftkey.py` |
| Students, transfer, readouts | `students/zz_leak_offsft16q.py`, `common/zz_sweep_lib.py`, `students/zz_mc8mat.py`, `students/zz_e35_sum.py` |
| Resampling and per-token log-likelihood ratio | `students/zz_rv_mkbank.py`, `students/zz_rv_anTD.py` |

## Data

`data/` holds the numbers behind every figure and table (`data/README.md`): the analysis outputs the figures
are drawn from (`data/raw`), their per-figure JSON exports (`data/figures`), the results of the additional
experiments (`data/revision`) and the teacher training data (`data/teacher_training`). Model weights
(fine-tuned teachers and students), activation dumps and the full generated-answer sets are not included;
they are available from the corresponding author on reasonable request.

## Citation

Kim, K. & Choi, Y. S. A balloon effect: ban one channel and a hidden trait finds another. Manuscript
submitted (2026).

Contact: Yong Suk Choi (cys@hanyang.ac.kr), Keuntae Kim (ktkpv94@hanyang.ac.kr).

## License

MIT (see `LICENSE`).
