# ukbFound: A Foundation Model for Deep Phenotyping

A lightweight yet rigorous foundation model that encodes thousands of UK Biobank (UKB) traits into language-like sequences for three families of applications: (1) disease subgroup stratification with survival differences; (2) inter‑disease correlation/community discovery; and (3) lifestyle‑based disease risk prediction.

> This repo contains: training code (`pre-train.py`), Python package modules under `ukbfound/`, and four reproducible notebooks for tokenization and downstream tasks.

This is the official codebase for **A foundational model encodes deep phenotyping data and enables diverse downstream application**. 

ukbFound: a foundation model with 25.3 million parameters that encoded thousands of individual-level traits into language-like sequences. By incorporating domain-specific tokenization, position-free embedding, and interpretable reasoning, ukbFound effectively captures latent disease-trait relationships from deep phenotyping data of 502,118 UK Biobank individuals.


![FIG1_20250502](https://github.com/user-attachments/assets/5194c3c2-dda1-488e-a002-3154e34e4424)


---

## Highlights
- **Hierarchical tokenization** for mixed data types
  - *Continuous* → quartiles (Q1–Q4).
  - *Multi‑choice* → trait token + choice value tokens.
  - *Multi‑select* → each choice becomes a binary trait (yes/no).
- **Position‑free input embedding**: each item is represented by the sum of its *trait* and *value* embeddings, removing dependence on column order.
- **Pretraining objective**: masked language modeling (MLM) with stratified masking across token types.
- **Interpretable downstream**: survival analysis for subgroups; cosine‑similarity graphs + Leiden clustering for multimorbidity; SHAP for feature attribution in prediction.

---

## Repository structure
```
ukbFound/
├─ pre-train.py                 # main pretraining script
├─ ukbfound/                    # library package
│  ├─ model.py                  # TransformerModel implementation
│  ├─ tokenizer/                # tokenization utilities (e.g., ValueVocab)
│  ├─ preprocess.py             # Preprocessor for binning/normalization
│  ├─ loss.py                   # MLM and related losses
│  ├─ utils/                    # seeds, logging, metrics
│  └─ trainer.py                # training implementation
├─ tutorials/
│  ├─ UKB-tokenization.ipynb    # build trait/value vocab & tokenized matrices
│  ├─ App1.stratification.ipynb # subgrouping + KM & Cox models
│  ├─ App2.correlation.ipynb    # disease–disease correlation & communities
│  └─ APP3.prediction.ipynb     # lifestyle-only disease prediction + SHAP
└─ data/
   ├─ output_data.csv        # tokenization input matrix (example path)
   └─ ukb_traits.csv         # trait↔token mapping (example path)
```

> **Note**: File/dir names above reflect typical usage in this repo. Adjust paths as needed in scripts/notebooks.

---

## Installation
Tested with **Python ≥ 3.8** and **PyTorch ≥ 1.13** (2.x recommended). Create a clean env and install the core deps:

```bash
conda create -n ukbfound python=3.10 -y
conda activate ukbfound

# PyTorch (pick the correct CUDA build for your system)
# See https://pytorch.org for the latest install command.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core scientific stack
pip install numpy pandas scikit-learn matplotlib seaborn

# Bio / tokenization / analysis helpers
pip install scanpy torchtext igraph networkx shap lifelines

# Logging (optional)
pip install wandb
```

**Optional acceleration.** Flash‑attention can be installed for speed‑ups on compatible GPUs/CUDA. It’s optional; if unavailable, the code falls back to standard attention.

---

## Quick start
### 1) Prepare data
Place UKB‑derived tabular features in `data/UKB/output_data.csv` (rows = individuals; columns = traits). Put the trait ↔ token mapping at `data/UKB/ukb_traits.csv`. The tokenization notebook shows how to generate these from raw UKB fields.

### 2) Pretrain ukbFound
`pre-train.py` exposes all key hyperparameters via a `hyperparameter_defaults` dict at the top of the script. Edit there for epochs, batch size, ECS threshold, etc.

**Single machine, multi‑GPU (recommended):**
```bash
# 4 GPUs example
torchrun --standalone --nproc_per_node=4 pre-train.py
```

**Common overrides inside `hyperparameter_defaults`:**
```python
epochs=50, batch_size=40, lr=1e-4,
layer_size=256, nlayers=4, nhead=4, dropout=0.2,
mask_ratio=0.15, n_bins=2,
fast_transformer=True, amp=True, pre_norm=False,
include_zero_trait=True, ecs_thres=0.0, DSBN=False,
```

**Data layout expected by the script**
- `data/UKB/output_data.csv` → main feature matrix.
- `data/UKB/ukb_traits.csv`    → trait/value vocabularies.

During a run, artifacts (args, vocab, checkpoints, logs) are written under `./save/dev_UKB-<timestamp>/`.

### 3) Downstream notebooks
Run the notebooks in order (they’re self‑contained with clear cells):
1. **UKB-tokenization.ipynb** — build trait and value vocabularies; export tokenized matrices.
2. **App1.stratification.ipynb** — embed disease cohorts, Leiden clustering, KM curves, and age‑adjusted Cox models.
3. **App2.correlation.ipynb** — cosine similarity of disease embeddings, multimorbidity pairs, and system‑level communities.
4. **APP3.prediction.ipynb** — train classifier on lifestyle/diet embeddings; evaluate AUC; inspect SHAP attributions.

> Tip: for batch runs, you can execute notebooks with `papermill` or convert them to `.py`.

---

## Conceptual design
- **Token vocabularies**
  - *Trait vocab*: one token per trait (e.g., height, smoking frequency, ICD‑10 disease presence, etc.).
  - *Value vocab*: quartiles for continuous traits; category choices for multi‑choice; yes/no for multi‑select.
- **Input representation**: per item embedding = `emb_trait + emb_value`. Pad/mask tokens are supported; disease tokens can be over‑sampled in masking to avoid bias.
- **Encoder**: Transformer with ~25M parameters (configurable depth/width). CLS/mean‑pooling for individual embedding.
- **Objectives**: MLM by default; optional ECS/DAB; AMP enabled by default.
- **Distributed**: DDP with `find_unused_parameters=True`; LOCAL_RANK handled automatically when using `torchrun`.

---

## Reproducibility
- Deterministic seeds via `set_seed`.
- Train/val split performed with `train_test_split(test_size=0.1, shuffle=True)`.
- Logging: W&B (offline mode by default). A full `run.log` is saved alongside checkpoints.

---

## Evaluation & expected results (reference)
The included notebooks recreate the key analyses:
- **Subgroup stratification**: prognostically distinct subgroups in many diseases using KM/log‑rank; Cox with baseline age as covariate.
- **Disease correlation**: tens of thousands of multimorbidity pairs; communities within systems (e.g., respiratory and cardiovascular) via cosine similarity + Leiden clustering.
- **Prediction (lifestyle‑only)**: robust AUCs across 100+ diseases; SHAP reveals interpretable risk/protective patterns; longitudinal risk stratification by baseline scores.

> Exact numbers will vary by cohort filters and seeds. See notebooks for figures/tables and to regenerate cohort‑specific metrics.

---

## Configuration reference
Key flags in `hyperparameter_defaults`:
- **Data**: `dataset_name`, `n_bins`, `include_zero_trait`.
- **Optimization**: `lr`, `batch_size`, `epochs`, `schedule_ratio`.
- **Architecture**: `layer_size (embsize & d_hid)`, `nlayers`, `nhead`, `dropout`, `pre_norm`.
- **Training tricks**: `mask_ratio`, `fast_transformer`, `amp`, `ecs_thres`, `DSBN`, `freeze`.

To resume from a checkpoint, set `load_model` to the checkpoint dir (it expects `args.json`, `best_model.pt`, and `vocab.json`).

---

## Data availability & ethics
- Individual‑level UKB data require an approved project and must comply with the UKB data access policy. This repo contains **no** raw UKB data. Token vocabularies and model inputs derived from UKB should be handled per UKB’s data‑sharing rules.

---

## Citation
If you use ukbFound, please cite the manuscript and this repository. *(Preprint/manuscript info placeholder here; update once public DOI is available.)*

```
@software{ukbfound_2025,
  title        = {ukbFound: A Foundation Model for Deep Phenotyping},
  author       = {Qiyang Hong and collaborators},
  year         = {2025},
  url          = {https://github.com/qiyanghong2020/ukbFound}
}
```

---

## Acknowledgements
This work was supported by national and institutional funding. We thank collaborators and computing support teams. See manuscript for full acknowledgements.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

---

## FAQ
**Q: My GPU doesn’t support flash‑attention. Can I still run it?**  
A: Yes. Set `fast_transformer=False` or install standard PyTorch builds; training will use regular attention.

**Q: How do I run on a single GPU/CPU?**  
A: Launch `python pre-train.py` (instead of `torchrun`) and reduce `batch_size`. Training will be slower.

**Q: How do I change which traits are included?**  
A: Edit the tokenization notebook to filter columns before building vocab; regenerate `output_data.csv` and restart training.


