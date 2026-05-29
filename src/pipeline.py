from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import umap

from models.autoencoder import AutoEncoder
from models.vae import VariationalAutoEncoder, vae_loss


@dataclass
class Challenge6Config:
    ch5_results_dir: Path
    output_dir: Path
    figures_dir: Path
    weights_dir: Path
    random_seeds: tuple[int, ...] = (7, 42, 1337)
    latent_dims: tuple[int, ...] = (8, 16)
    betas: tuple[float, ...] = (0.5, 1.0, 4.0)
    hidden_dims: tuple[int, ...] = (128, 64)
    test_size: float = 0.2
    batch_size: int = 512
    ae_epochs: int = 70
    vae_epochs: int = 90
    learning_rate: float = 1e-3
    train_without_top_iforest: float = 0.05
    anomaly_percentile: float = 95.0
    max_samples: int | None = 30000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_ch5_artifacts(config: Challenge6Config) -> tuple[np.ndarray, np.ndarray]:
    x_path = config.ch5_results_dir / "X_processed.npy"
    kmeans_path = config.ch5_results_dir / "kmeans_labels.npy"

    if not x_path.exists() or not kmeans_path.exists():
        raise FileNotFoundError(
            "Required Challenge 5 files were not found. Expected X_processed.npy and kmeans_labels.npy"
        )

    x = np.load(x_path)
    kmeans_labels = np.load(kmeans_path)

    if config.max_samples is not None and x.shape[0] > config.max_samples:
        idx = np.random.default_rng(42).choice(x.shape[0], size=config.max_samples, replace=False)
        x = x[idx]
        kmeans_labels = kmeans_labels[idx]

    return x.astype(np.float32), kmeans_labels


def build_loader(x_np: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor = torch.tensor(x_np, dtype=torch.float32)
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_autoencoder(
    x_train: np.ndarray,
    config: Challenge6Config,
    seed: int,
    latent_dim: int,
) -> tuple[AutoEncoder, list[float]]:
    seed_everything(seed)

    model = AutoEncoder(
        input_dim=x_train.shape[1],
        hidden_dims=list(config.hidden_dims),
        latent_dim=latent_dim,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    loader = build_loader(x_train, config.batch_size, shuffle=True)

    losses: list[float] = []
    for _ in range(config.ae_epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(config.device)
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = batch_x.size(0)
            epoch_loss += float(loss.item()) * batch_n
            n += batch_n

        losses.append(epoch_loss / max(n, 1))

    return model, losses


def infer_autoencoder(model: AutoEncoder, x_np: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_hat, z = model(x_t)
        errors = torch.mean((x_t - x_hat) ** 2, dim=1).cpu().numpy()
        latent = z.cpu().numpy()
    return errors, latent


def train_vae(
    x_train: np.ndarray,
    config: Challenge6Config,
    seed: int,
    latent_dim: int,
    beta: float,
) -> tuple[VariationalAutoEncoder, list[float], list[float], list[float]]:
    seed_everything(seed)

    model = VariationalAutoEncoder(
        input_dim=x_train.shape[1],
        hidden_dims=list(config.hidden_dims),
        latent_dim=latent_dim,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = build_loader(x_train, config.batch_size, shuffle=True)

    total_losses: list[float] = []
    recon_losses: list[float] = []
    kl_losses: list[float] = []

    for _ in range(config.vae_epochs):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(config.device)
            x_hat, mu, logvar = model(batch_x)
            total, recon, kl = vae_loss(x_hat, batch_x, mu, logvar, beta=beta)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            batch_n = batch_x.size(0)
            epoch_total += float(total.item()) * batch_n
            epoch_recon += float(recon.item()) * batch_n
            epoch_kl += float(kl.item()) * batch_n
            n += batch_n

        n = max(n, 1)
        total_losses.append(epoch_total / n)
        recon_losses.append(epoch_recon / n)
        kl_losses.append(epoch_kl / n)

    return model, total_losses, recon_losses, kl_losses


def infer_vae(
    model: VariationalAutoEncoder,
    x_np: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_hat, mu, _ = model(x_t)
        errors = torch.mean((x_t - x_hat) ** 2, dim=1).cpu().numpy()
        mu_np = mu.cpu().numpy()
    return errors, mu_np


def fit_isolation_forest(
    x_train: np.ndarray,
    x_all: np.ndarray,
    seed: int,
) -> np.ndarray:
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(x_train)
    return -iso.score_samples(x_all)


def plot_ae_loss(losses: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(losses, linewidth=2)
    ax.set_title("AE Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_vae_loss(total: list[float], recon: list[float], kl: list[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(total, label="total", linewidth=2)
    ax.plot(recon, label="reconstruction", linewidth=2)
    ax.plot(kl, label="kl", linewidth=2)
    ax.set_title("VAE Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_hist_with_threshold(errors: np.ndarray, threshold: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(errors, bins=50, kde=True, ax=ax)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
    ax.set_title("AE Reconstruction Error Distribution")
    ax.set_xlabel("Per-sample MSE")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_tsne_vae(mu: np.ndarray, anomaly_score: np.ndarray, save_path: Path) -> None:
    max_points = 8000
    rng = np.random.default_rng(42)
    idx = rng.choice(mu.shape[0], size=min(max_points, mu.shape[0]), replace=False)

    z = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(mu[idx])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z[:, 0], z[:, 1], c=anomaly_score[idx], cmap="plasma", s=12, alpha=0.8)
    ax.set_title("t-SNE of VAE latent means (colored by VAE error)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("VAE reconstruction error")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_umap_ae(latent: np.ndarray, cluster_labels: np.ndarray, save_path: Path) -> None:
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, random_state=42)
    z2 = reducer.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=cluster_labels, cmap="tab10", s=12, alpha=0.8)
    ax.set_title("UMAP of AE latent vectors (colored by Challenge 5 KMeans labels)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    fig.colorbar(sc, ax=ax, label="KMeans label")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_ae_vs_iso(ae_error: np.ndarray, iso_score: np.ndarray, rho: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ae_error, iso_score, s=12, alpha=0.35)
    ax.set_title(f"AE error vs Isolation Forest score (Spearman rho={rho:.3f})")
    ax.set_xlabel("AE reconstruction error")
    ax.set_ylabel("Isolation Forest score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_top_anomalies(
    metric_name: str,
    scores: np.ndarray,
    output_path: Path,
    top_n: int = 10,
) -> None:
    top_idx = np.argsort(scores)[::-1][:top_n]
    df = pd.DataFrame({"rank": np.arange(1, top_n + 1), "sample_index": top_idx, metric_name: scores[top_idx]})
    df.to_csv(output_path, index=False)


def split_train_eval(
    x: np.ndarray,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    return x[idx_train], x[idx_test], idx_train, idx_test


def remove_potential_anomalies(x_train: np.ndarray, seed: int, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return x_train

    iso = IsolationForest(n_estimators=200, contamination=fraction, random_state=seed, n_jobs=-1)
    labels = iso.fit_predict(x_train)
    keep = labels == 1
    if keep.sum() < 100:
        return x_train
    return x_train[keep]


def run(config: Challenge6Config) -> None:
    ensure_dirs(config.output_dir, config.figures_dir, config.weights_dir)
    x, ch5_kmeans = load_ch5_artifacts(config)

    all_rows: list[dict] = []
    best_bundle: dict | None = None

    for seed in config.random_seeds:
        x_train_full, _, _, _ = split_train_eval(x, seed=seed, test_size=config.test_size)
        x_train = remove_potential_anomalies(
            x_train_full,
            seed=seed,
            fraction=config.train_without_top_iforest,
        )

        iso_scores = fit_isolation_forest(x_train, x, seed=seed)
        iso_thr = np.percentile(iso_scores, config.anomaly_percentile)
        iso_rate = float((iso_scores > iso_thr).mean())

        for latent_dim in config.latent_dims:
            ae_model, ae_losses = train_autoencoder(x_train, config, seed=seed, latent_dim=latent_dim)
            ae_error, ae_latent = infer_autoencoder(ae_model, x, config.device)
            ae_thr = np.percentile(ae_error, config.anomaly_percentile)
            ae_rate = float((ae_error > ae_thr).mean())

            sil_raw = float(silhouette_score(x, ch5_kmeans))
            sil_ae = float(silhouette_score(ae_latent, ch5_kmeans))
            rho_ae_iso = float(spearmanr(ae_error, iso_scores).statistic)

            row_ae = {
                "model": "AE",
                "seed": seed,
                "latent_dim": latent_dim,
                "beta": np.nan,
                "final_train_loss": ae_losses[-1],
                "anomaly_rate": ae_rate,
                "spearman_vs_iso": rho_ae_iso,
                "silhouette_raw_with_ch5": sil_raw,
                "silhouette_latent_with_ch5": sil_ae,
            }
            all_rows.append(row_ae)

            torch.save(
                ae_model.state_dict(),
                config.weights_dir / f"ae_seed{seed}_ld{latent_dim}.pt",
            )

            for beta in config.betas:
                vae_model, vae_total, vae_recon, vae_kl = train_vae(
                    x_train,
                    config,
                    seed=seed,
                    latent_dim=latent_dim,
                    beta=beta,
                )
                vae_error, vae_mu = infer_vae(vae_model, x, config.device)
                vae_thr = np.percentile(vae_error, config.anomaly_percentile)
                vae_rate = float((vae_error > vae_thr).mean())

                sil_vae = float(silhouette_score(vae_mu, ch5_kmeans))
                rho_vae_iso = float(spearmanr(vae_error, iso_scores).statistic)

                row_vae = {
                    "model": "VAE",
                    "seed": seed,
                    "latent_dim": latent_dim,
                    "beta": beta,
                    "final_train_loss": vae_total[-1],
                    "anomaly_rate": vae_rate,
                    "spearman_vs_iso": rho_vae_iso,
                    "silhouette_raw_with_ch5": sil_raw,
                    "silhouette_latent_with_ch5": sil_vae,
                }
                all_rows.append(row_vae)

                torch.save(
                    vae_model.state_dict(),
                    config.weights_dir / f"vae_seed{seed}_ld{latent_dim}_b{beta}.pt",
                )

                score = sil_vae + rho_vae_iso
                if best_bundle is None or score > best_bundle["score"]:
                    best_bundle = {
                        "score": score,
                        "seed": seed,
                        "latent_dim": latent_dim,
                        "beta": beta,
                        "ae_model": ae_model,
                        "ae_losses": ae_losses,
                        "ae_error": ae_error,
                        "ae_latent": ae_latent,
                        "vae_model": vae_model,
                        "vae_total": vae_total,
                        "vae_recon": vae_recon,
                        "vae_kl": vae_kl,
                        "vae_error": vae_error,
                        "vae_mu": vae_mu,
                        "iso_scores": iso_scores,
                        "iso_rate": iso_rate,
                        "ch5_kmeans": ch5_kmeans,
                    }

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(config.output_dir / "metrics_by_seed.csv", index=False)

    summary = (
        metrics.groupby(["model", "latent_dim", "beta"], dropna=False)
        .agg(
            final_train_loss_mean=("final_train_loss", "mean"),
            final_train_loss_std=("final_train_loss", "std"),
            anomaly_rate_mean=("anomaly_rate", "mean"),
            anomaly_rate_std=("anomaly_rate", "std"),
            spearman_vs_iso_mean=("spearman_vs_iso", "mean"),
            spearman_vs_iso_std=("spearman_vs_iso", "std"),
            silhouette_latent_mean=("silhouette_latent_with_ch5", "mean"),
            silhouette_latent_std=("silhouette_latent_with_ch5", "std"),
        )
        .reset_index()
    )
    summary.to_csv(config.output_dir / "summary_table.csv", index=False)

    baseline = pd.DataFrame(
        [{"model": "IsolationForest", "anomaly_rate": best_bundle["iso_rate"], "seed": best_bundle["seed"]}]
    )
    baseline.to_csv(config.output_dir / "iforest_baseline.csv", index=False)

    save_top_anomalies(
        metric_name="ae_error",
        scores=best_bundle["ae_error"],
        output_path=config.output_dir / "top10_ae_anomalies.csv",
    )
    save_top_anomalies(
        metric_name="vae_error",
        scores=best_bundle["vae_error"],
        output_path=config.output_dir / "top10_vae_anomalies.csv",
    )
    save_top_anomalies(
        metric_name="iforest_score",
        scores=best_bundle["iso_scores"],
        output_path=config.output_dir / "top10_iforest_anomalies.csv",
    )

    ae_thr = np.percentile(best_bundle["ae_error"], config.anomaly_percentile)
    rho = float(spearmanr(best_bundle["ae_error"], best_bundle["iso_scores"]).statistic)

    plot_ae_loss(best_bundle["ae_losses"], config.figures_dir / "ae_training_loss.png")
    plot_vae_loss(
        best_bundle["vae_total"],
        best_bundle["vae_recon"],
        best_bundle["vae_kl"],
        config.figures_dir / "vae_training_loss.png",
    )
    plot_hist_with_threshold(
        best_bundle["ae_error"],
        ae_thr,
        config.figures_dir / "ae_error_hist_threshold.png",
    )
    plot_tsne_vae(
        best_bundle["vae_mu"],
        best_bundle["vae_error"],
        config.figures_dir / "vae_tsne_anomaly_score.png",
    )
    plot_umap_ae(
        best_bundle["ae_latent"],
        best_bundle["ch5_kmeans"],
        config.figures_dir / "ae_umap_ch5_labels.png",
    )
    plot_ae_vs_iso(
        best_bundle["ae_error"],
        best_bundle["iso_scores"],
        rho,
        config.figures_dir / "ae_vs_iforest_scatter.png",
    )

    best_config = {
        "seed": int(best_bundle["seed"]),
        "latent_dim": int(best_bundle["latent_dim"]),
        "beta": float(best_bundle["beta"]),
        "device": config.device,
        "random_seeds": list(config.random_seeds),
        "latent_dims": list(config.latent_dims),
        "betas": list(config.betas),
    }
    with open(config.output_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    print("Challenge 6 pipeline completed.")
    print(f"Best config: {best_config}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Challenge 6 Group 3 pipeline")
    parser.add_argument(
        "--ch5-results-dir",
        type=str,
        default="../results",
        help="Path to Challenge 5 results folder containing X_processed.npy and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory where CSV outputs are saved.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="./figures",
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="./weights",
        help="Directory where model checkpoints are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Challenge6Config(
        ch5_results_dir=Path(args.ch5_results_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        figures_dir=Path(args.figures_dir).resolve(),
        weights_dir=Path(args.weights_dir).resolve(),
    )

    run(config)


if __name__ == "__main__":
    main()
