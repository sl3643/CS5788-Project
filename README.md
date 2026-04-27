# Platonic Language Representations

This repository implements a language-only empirical study of the Platonic Representation Hypothesis. 
We compare sentence-level hidden representations from several pretrained generative language models using layer-wise CKA, 
nearest-neighbor overlap, SVCCA, and PCA/UMAP visualizations.

## Project structure

- `src/`: main implementation
- `scripts/`: command-line entry points
- `data/`: processed text dataset
- `outputs/`: metrics and figures
- `notebooks/`: minimal demo notebook
- `report/`: final CVPR-style report files

## Setup

```bash
pip install -r requirements.txt