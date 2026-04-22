import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_covariance(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "cov" not in data:
        raise KeyError(f"'cov' not found in {npz_path}")
    cov = np.asarray(data["cov"], dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got shape {cov.shape}")
    return cov


def downsample_matrix(mat: np.ndarray, max_side: int = 256) -> np.ndarray:
    n = mat.shape[0]
    if n <= max_side:
        return mat

    block = int(np.ceil(n / max_side))
    pad = block * max_side - n
    if pad > 0:
        mat = np.pad(mat, ((0, pad), (0, pad)), mode="constant", constant_values=np.nan)

    m = mat.shape[0] // block
    mat = mat.reshape(m, block, m, block)
    return np.nanmean(mat, axis=(1, 3))


def plot_covariance(cov: np.ndarray, output_dir: Path, title_prefix: str, max_side: int = 256) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    diag = np.diag(cov)
    std = np.sqrt(np.clip(diag, 0.0, None))

    cov_ds = downsample_matrix(cov, max_side=max_side)
    vmax = np.nanpercentile(np.abs(cov_ds), 99)
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else None

    plt.close("all")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im = ax1.imshow(cov_ds, cmap="coolwarm", vmin=-vmax if vmax is not None else None, vmax=vmax)
    ax1.set_title(f"{title_prefix} Covariance Heatmap (downsampled)")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Dimension")
    fig1.colorbar(im, ax=ax1, shrink=0.8)
    fig1.tight_layout()
    heatmap_path = output_dir / "covariance_heatmap.png"
    fig1.savefig(heatmap_path, dpi=200)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(diag, label="diag(cov)")
    ax2.set_title(f"{title_prefix} Covariance Diagonal")
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Variance")
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    diag_path = output_dir / "covariance_diagonal.png"
    fig2.savefig(diag_path, dpi=200)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(std, label="sqrt(diag(cov))", color="tab:orange")
    ax3.set_title(f"{title_prefix} Standard Deviation Per Dimension")
    ax3.set_xlabel("Dimension")
    ax3.set_ylabel("Std")
    ax3.grid(True, linestyle="--", alpha=0.5)
    fig3.tight_layout()
    std_path = output_dir / "covariance_std.png"
    fig3.savefig(std_path, dpi=200)

    print(f"Saved covariance heatmap to {heatmap_path}")
    print(f"Saved covariance diagonal plot to {diag_path}")
    print(f"Saved covariance std plot to {std_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a saved covariance matrix from a solver_state npz file."
    )
    parser.add_argument("npz_path", type=Path, help="Path to solver_state_final.npz or similar file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to <npz parent>/covariance_viz",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="CEM",
        help="Prefix used in plot titles",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=256,
        help="Maximum heatmap side length after block downsampling",
    )
    args = parser.parse_args()

    cov = load_covariance(args.npz_path)
    output_dir = args.output_dir if args.output_dir is not None else args.npz_path.parent / "covariance_viz"
    plot_covariance(cov, output_dir, args.title_prefix, max_side=args.max_side)


if __name__ == "__main__":
    main()
