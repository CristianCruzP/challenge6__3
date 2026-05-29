# Challenge 6 Checklist — Group 3

## Dataset
- Domain: Public Health & Epidemiology
- Dataset: BRFSS sample reused from Challenge 5
- Source files: `data/brfss_sample_30k.csv`, `data/brfss_sample_5k.csv`
- Challenge 5 processed matrix: `results/challenge5/X_processed.npy`
- Challenge 5 K-Means labels: `results/challenge5/kmeans_labels.npy`

## Models
- AE architecture: defined in `src/models/autoencoder.py` and trained by `src/pipeline.py`
- VAE architecture: defined in `src/models/vae.py` and trained by `src/pipeline.py`
- Isolation Forest baseline: trained in `src/pipeline.py`

## Reproducibility
- Random seeds: `7`, `42`, `1337`
- Output folders:
  - `results/challenge6`
  - `figures/challenge6`
  - `weights/challenge6`
- Main runnable notebook: `notebooks/challenge6/01_group3_challenge6.ipynb`
- Main script: `src/pipeline.py`

## Fill after running the pipeline
- AE latent dimension:
- VAE latent dimension:
- VAE beta:
- AE anomaly threshold:
- VAE anomaly threshold:
- Isolation Forest threshold:
- AE anomaly rate:
- VAE anomaly rate:
- Isolation Forest anomaly rate:
- Spearman correlation AE vs Isolation Forest:
- Spearman correlation VAE vs Isolation Forest:
- Silhouette score in raw space:
- Silhouette score in AE latent space:
- Silhouette score in VAE latent space:

## Short synthesis
- What Challenge 6 adds beyond Challenges 2 and 5 will be written here after the pipeline results are finalized.
