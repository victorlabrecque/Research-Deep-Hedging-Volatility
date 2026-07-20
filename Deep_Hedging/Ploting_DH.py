import matplotlib.pyplot as plt
import numpy as np


# ____________________________________

# --- 1. Plot n paths ---


def plot_n_paths(paths, n, K, T, m=0):
    """
    Plot the first n paths of the simulated stock prices
    Paths are ranked by terminal moneyness
    """

    first_n = paths[m : n + m, :]
    sort_idx = np.argsort(first_n[:, -1])[::-1]
    first_n = first_n[sort_idx, :]

    time_axis = np.linspace(0, T, first_n.shape[1])
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / (n - 1)) for i in range(n - 1, -1, -1)]

    plt.figure(figsize=(10, 4))
    for i, (path, color) in enumerate(zip(first_n, colors)):
        moneyness = path[-1] / K
        plt.plot(
            time_axis,
            path,
            color=color,
            label=f"Path {i + 1} | Moneyness={moneyness:.2f}",
        )

    plt.axhline(K, color="black", linestyle="--", linewidth=1, label="Strike K")
    plt.title(f"First {n} Simulated Paths")
    plt.xlabel("Time")
    plt.ylabel("$S_t$")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid()
    plt.tight_layout()
    plt.show()


# ____________________________________

# --- 2. Histogram of 2 pnl ---


def plot_pnl_histogram(
    pnl1, pnl2, label1="P&L 1", label2="P&L 2", title="P&L Distribution", bins=50
):
    plt.figure(figsize=(8, 4))
    plt.hist(pnl1, bins=bins, density=True, alpha=0.5, label=label1)
    plt.hist(pnl2, bins=bins, density=True, alpha=0.5, label=label2)
    plt.title(title)
    plt.xlabel("P&L")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# ____________________________________

# --- 3. Compare 2 deltas ---


def plot_n_deltas(
    paths, deltas1, deltas2, n, K, T, m=0, name1="deltas1", name2="deltas2"
):
    """
    1.Plot the first n paths deltas of the simulated stock prices
    2.Compare deltas
    """

    # --- Sort paths by terminal moneyness, then filter on deltas ---
    first_n = paths[m : n + m, :]
    sort_idx = np.argsort(first_n[:, -1])[::-1]
    first_n = first_n[sort_idx, :]
    terminal_moneyness = first_n[:, -1] / K
    deltas1_n = deltas1[sort_idx, :]
    deltas2_n = deltas2[sort_idx, :]

    # time_axis = np.linspace(0, T, first_n.shape[1])

    ytick_labels = [f"{terminal_moneyness[i]:.3f}" for i in range(n)]

    # --- Heatmap: deltas1 and deltas2 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im1 = axes[0].imshow(
        deltas1_n, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="upper"
    )
    axes[0].set_title(name1)
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Terminal Moneyness $S_T/K$")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(ytick_labels)
    plt.colorbar(im1, ax=axes[0], label="Delta")

    im2 = axes[1].imshow(
        deltas2_n, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="upper"
    )
    axes[1].set_title(name2)
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Terminal Moneyness $S_T/K$")
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels(ytick_labels)
    plt.colorbar(im2, ax=axes[1], label="Delta")

    plt.suptitle(
        f"Delta Heatmaps - {n} Paths sorted by Terminal Moneyness", fontsize=13
    )
    plt.tight_layout()
    plt.show()

    # --- Plot Deltas difference ---

    # Green: deltas1 > deltas2 | Red: deltas1 < deltas2
    deltas_diff = deltas1_n - deltas2_n
    fig, ax = plt.subplots(figsize=(16, 5))
    im3 = ax.imshow(
        deltas_diff, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1, origin="upper"
    )
    ax.set_title(f"{name1} - {name2}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Terminal Moneyness $S_T/K$")
    ax.set_yticks(range(n))
    ax.set_yticklabels(ytick_labels)
    plt.colorbar(
        im3,
        ax=ax,
        label="Delta Difference (green: deltas1 > deltas2 | red: deltas1 < deltas2)",
    )
    plt.suptitle(
        f"Delta Difference — {n} Paths sorted by Terminal Moneyness (ITM → OTM)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()

    # --- 4. Correlation ---
    def plot_correlation(delta1, delta2, T, label1="delta1", label2="delta2"):
        delta1_c = delta1 - delta1.mean(axis=0)
        delta2_c = delta2 - delta2.mean(axis=0)
        num = (delta1_c * delta2_c).sum(axis=0)
        denum = np.sqrt((delta1_c**2).sum(axis=0) * (delta2_c**2).sum(axis=0))
        corr = num / denum

        time_axis = np.linspace(0, T, len(corr))

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, corr, color="steelblue", linewidth=1.5)
        plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.axhline(1, color="gray", linewidth=0.8, linestyle=":")
        plt.axhline(-1, color="gray", linewidth=0.8, linestyle=":")
        plt.fill_between(time_axis, corr, 0, alpha=0.15, color="steelblue")
        plt.ylim(-1.05, 1.05)
        plt.xlabel("Time")
        plt.ylabel("Correlation")
        plt.title(f"Delta Correlation: {label1} vs {label2}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
