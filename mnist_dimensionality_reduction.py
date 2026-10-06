"""MNIST dimensionality reduction and recognition analysis.

Downloads the CSV files linked in the assignment, performs PCA and t-SNE, and
compares 5-NN classification on raw and PCA-transformed pixels.

Example:
    python mnist_dimensionality_reduction.py
"""

from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


TRAIN_URL = "https://moovit.vit.ac.in/pluginfile.php/1113531/mod_assign/intro/mnist_train.csv"
TEST_URL = "https://moovit.vit.ac.in/pluginfile.php/1113531/mod_assign/intro/mnist_test.csv"


def fetch_csv(url: str, destination: Path) -> Path:
    """Download a CSV if it is not already present locally."""
    if not destination.exists():
        print(f"Downloading {url}\n        -> {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, destination)
    return destination


def load_mnist_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a label-first MNIST CSV, including files with or without headers."""
    frame = pd.read_csv(path, header=None)
    # The provided assignment CSVs are headerless. If a header exists, discard
    # it by coercing all values to numeric and dropping the non-numeric row.
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if frame.shape[1] != 785:
        raise ValueError(f"Expected 785 columns (label + 784 pixels) in {path}; got {frame.shape[1]}.")
    values = frame.to_numpy(dtype=np.float32)
    labels = values[:, 0].astype(np.int64)
    pixels = values[:, 1:]
    if pixels.shape[1] != 784:
        raise ValueError("MNIST images must contain 784 pixel columns.")
    return pixels / 255.0, labels


def save_sample_grid(X: np.ndarray, y: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")
    fig.suptitle("Sample MNIST handwritten digits")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_pca_plots(pca: PCA, X_2d: np.ndarray, y: np.ndarray, output_dir: Path) -> None:
    ratios = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.arange(1, len(ratios) + 1), ratios, linewidth=1)
    axes[0].set(title="PCA Scree Plot", xlabel="Principal component", ylabel="Variance explained")
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), linewidth=1.5)
    axes[1].axhline(.90, color="orange", linestyle="--", label="90%")
    axes[1].axhline(.95, color="red", linestyle="--", label="95%")
    axes[1].axhline(.99, color="green", linestyle="--", label="99%")
    axes[1].set(title="Cumulative Explained Variance", xlabel="Number of components", ylabel="Cumulative variance")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pca_variance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=5, alpha=.65)
    ax.set(title="MNIST: 2D PCA projection", xlabel="PC 1", ylabel="PC 2")
    fig.colorbar(points, ax=ax, ticks=range(10), label="Digit")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_2d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def tsne_kwargs(perplexity: float, seed: int) -> dict:
    params = inspect.signature(TSNE).parameters
    iteration_key = "max_iter" if "max_iter" in params else "n_iter"
    return {"n_components": 2, "perplexity": perplexity, "init": "pca",
            "learning_rate": "auto", "random_state": seed, iteration_key: 1000}


def run_tsne(X_raw: np.ndarray, X_pca: np.ndarray, y: np.ndarray,
             perplexities: list[float], output_dir: Path, sample_count: int, seed: int) -> None:
    # t-SNE is cubic-ish in sample count; use a reproducible stratified subset.
    if sample_count < len(y):
        _, indices = train_test_split(np.arange(len(y)), train_size=sample_count,
                                      stratify=y, random_state=seed)
    else:
        indices = np.arange(len(y))
    labels = y[indices]
    raw = X_raw[indices]
    pca = X_pca[indices]
    print(f"\nt-SNE uses {len(indices):,} stratified training samples (out of {len(y):,}).")
    for perplexity in perplexities:
        if perplexity >= len(indices):
            print(f"Skipping perplexity {perplexity}: it must be smaller than sample count.")
            continue
        for name, data in (("original", raw), ("pca50", pca)):
            print(f"Fitting t-SNE on {name}, perplexity={perplexity}...")
            start = time.perf_counter()
            embedding = TSNE(**tsne_kwargs(perplexity, seed)).fit_transform(data)
            elapsed = time.perf_counter() - start
            fig, ax = plt.subplots(figsize=(9, 7))
            points = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels,
                                cmap="tab10", s=5, alpha=.65)
            ax.set(title=f"t-SNE ({name}, perplexity={perplexity}; {elapsed:.1f}s)",
                   xlabel="t-SNE 1", ylabel="t-SNE 2")
            fig.colorbar(points, ax=ax, ticks=range(10), label="Digit")
            fig.tight_layout()
            fig.savefig(output_dir / f"tsne_{name}_perplexity_{perplexity:g}.png",
                        dpi=160, bbox_inches="tight")
            plt.close(fig)


def evaluate_knn(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    accuracy = accuracy_score(y_test, predictions)
    print(f"{name}: accuracy={accuracy:.4f}, fit={train_seconds:.2f}s, "
          f"prediction={predict_seconds:.2f}s")
    return {"representation": name, "dimensions": X_train.shape[1], "accuracy": accuracy,
            "training_seconds": train_seconds, "prediction_seconds": predict_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory for downloaded MNIST CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path("mnist_results"), help="Directory for plots and CSV results")
    parser.add_argument("--tsne-samples", type=int, default=5000,
                        help="Stratified training subset for t-SNE (default: 5000; t-SNE is expensive)")
    parser.add_argument("--perplexities", type=float, nargs="+", default=[5, 30, 50],
                        help="Perplexities to compare (default: 5 30 50)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = fetch_csv(TRAIN_URL, args.data_dir / "mnist_train.csv")
    test_path = fetch_csv(TEST_URL, args.data_dir / "mnist_test.csv")
    X_train, y_train = load_mnist_csv(train_path)
    X_test, y_test = load_mnist_csv(test_path)

    print("PART A: DATA UNDERSTANDING AND PREPROCESSING")
    print(f"Training samples: {len(y_train):,}; test samples: {len(y_test):,}")
    print(f"Number of classes: {len(np.unique(y_train))}; labels: {np.unique(y_train)}")
    print(f"Image dimensions: 28 x 28 pixels ({X_train.shape[1]} features)")
    print("Training label counts:")
    labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"  {label}: {count:,}")
    print(f"Normalized pixel range: [{X_train.min():.1f}, {X_train.max():.1f}]")
    save_sample_grid(X_train, y_train, args.output_dir / "sample_digits.png")

    print("\nPART B: PCA")
    # Full SVD yields per-component variance and exact 90/95/99% thresholds.
    pca_all = PCA(svd_solver="full")
    start = time.perf_counter()
    pca_all.fit(X_train)
    print(f"PCA full fit time: {time.perf_counter() - start:.1f}s")
    cumulative = np.cumsum(pca_all.explained_variance_ratio_)
    for threshold in (.90, .95, .99):
        needed = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"Components for {threshold:.0%} variance: {needed}")
    pd.DataFrame({"component": np.arange(1, len(cumulative) + 1),
                  "explained_variance_ratio": pca_all.explained_variance_ratio_,
                  "cumulative_variance": cumulative}).to_csv(args.output_dir / "pca_variance.csv", index=False)

    pca_2 = PCA(n_components=2, svd_solver="full")
    X_train_2 = pca_2.fit_transform(X_train)
    pca_50 = PCA(n_components=50, svd_solver="full")
    X_train_50 = pca_50.fit_transform(X_train)
    X_test_50 = pca_50.transform(X_test)
    print(f"Variance retained by 2 components: {pca_2.explained_variance_ratio_.sum():.4f}")
    print(f"Variance retained by 50 components: {pca_50.explained_variance_ratio_.sum():.4f}")
    save_pca_plots(pca_all, X_train_2, y_train, args.output_dir)

    # Reconstruction example, as commonly requested in PCA assignments.
    reconstructed = pca_50.inverse_transform(X_train_50[:10])
    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(X_train[i].reshape(28, 28), cmap="gray")
        axes[1, i].imshow(np.clip(reconstructed[i].reshape(28, 28), 0, 1), cmap="gray")
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original"); axes[1, 0].set_ylabel("PCA-50")
    fig.suptitle("Image reconstruction with 50 principal components")
    fig.tight_layout()
    fig.savefig(args.output_dir / "pca50_reconstruction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nPART C: t-SNE")
    run_tsne(X_train, X_train_50, y_train, args.perplexities,
             args.output_dir, args.tsne_samples, args.seed)
    print("Compare cluster separation and overlap across each saved t-SNE plot. "
          "Perplexity changes the neighborhood scale; t-SNE plots preserve local neighborhoods, "
          "so global distances between clusters should not be interpreted literally.")

    print("\nPART D: DIGIT RECOGNITION (5-nearest neighbors)")
    results = [
        evaluate_knn(X_train, y_train, X_test, y_test, "Original 784D"),
        evaluate_knn(X_train_50, y_train, X_test_50, y_test, "PCA 50D"),
    ]
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(args.output_dir / "classification_results.csv", index=False)
    result_frame["training_memory_mb"] = [X_train.nbytes / 1024**2, X_train_50.nbytes / 1024**2]
    result_frame["prediction_memory_mb"] = [X_test.nbytes / 1024**2, X_test_50.nbytes / 1024**2]
    print("\nPerformance and feature-array memory:")
    print(result_frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInterpretation: PCA compresses the pixel representation and can lower the "
          "cost of distance calculations. Recognition accuracy may decrease if discarded "
          "components contain class-discriminative details; consult the measured results.")
    print(f"\nPlots and result tables saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
