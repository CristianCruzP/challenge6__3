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
- Best seed: 7
- AE latent dimension: 8
- VAE latent dimension: 8
- VAE beta: 0.5
- AE anomaly threshold: 0.0861
- VAE anomaly threshold: 0.6387
- Isolation Forest threshold: 0.5171
- AE anomaly rate: 0.0500
- VAE anomaly rate: 0.0500
- Isolation Forest anomaly rate: 0.0500
- Spearman correlation AE vs Isolation Forest: 0.7443
- Spearman correlation VAE vs Isolation Forest: 0.7266
- Silhouette score in raw space: 0.1433
- Silhouette score in AE latent space: 0.1871
- Silhouette score in VAE latent space: 0.2314

## Short synthesis
- Challenge 6 adds AE, VAE and Isolation Forest anomaly detection on the Challenge 5 processed matrix, plus t-SNE/UMAP views of the learned embeddings. The best run used seed 7 with latent dimension 8 and VAE beta 0.5; the latent spaces show better alignment with the Challenge 5 K-Means structure than the raw space, and the anomaly scores are strongly correlated with Isolation Forest.
