# Platonic Language Representations

This is the code repository for our CS5788 final project, A Language Study of the Platonic Representation Hypothesis

The project compares sentence-level hidden representations from several pretrained generative language models on the same set of WikiText-2 passages. We use layer-wise CKA, nearest-neighbor overlap, shuffled-text baselines, and PCA/UMAP visualizations to study whether different model architectures learn similar representation geometry

## Models

We compare four pretrained language models:

| Model | Checkpoint | Family |
|---|---|---|
| GPT-2 | `gpt2` | Decoder-only Transformer |
| Pythia-70M | `EleutherAI/pythia-70m` | Decoder-only Transformer |
| Mamba-130M | `state-spaces/mamba-130m-hf` | State-space model |
| RWKV-169M | `RWKV/rwkv-4-169m-pile` | Recurrent-like model |

## Repository structure

```text
src/          Main implementation
scripts/      Command-line scripts
notebooks/    Demo notebook
outputs/      Generated metrics and figures
data/         Processed text samples
report/       Final report files
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python -m src.run_all --config config.yaml
```

The same pipeline can also be run step by step:

```bash
python scripts/01_prepare_data.py --config config.yaml
python scripts/02_extract_all.py --config config.yaml
python scripts/03_compute_metrics.py --config config.yaml
python scripts/04_make_figures.py --config config.yaml
```

## Final experiment setting

The final experiment uses 500 filtered WikiText-2 passages, mean-pooled sentence representations, and all enabled models in `config.yaml`

Main outputs:

```text
outputs/metrics/summary_table.csv
outputs/metrics/cka_results.csv
outputs/metrics/nn_overlap_results.csv
outputs/figures/
```
