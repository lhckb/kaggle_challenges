import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.metrics import precision_recall_curve, roc_curve, auc

plt.rcParams["figure.dpi"] = 150
plt.rcParams["boxplot.medianprops.color"] = "black"

def histogram(data: pl.Series, title: str = "Title", mean_value: float = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    bins = len(data)//10000
    ax.hist(data, bins=bins)

    if mean_value > -1:
        ax.axvline(
            mean_value,
            linestyle='--', 
            alpha=0.7,
            color="grey",
            label="Mean"
        )
        plt.legend()

    plt.title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=10,
        loc="left"
    )
    plt.show()

def boxplot(data: pl.Series, title: str = "Title", mean_value: float = None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)

    _ = ax.boxplot(
        data,
        # vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", zorder=1),
        whiskerprops=dict(color="black", zorder=1),
        capprops=dict(color="black", zorder=1),
        medianprops=dict(color="black", linewidth=2, zorder=2)
    )

    if mean_value is not None:
        ax.axvline(
            mean_value,
            linestyle="--",
            color="grey",
            alpha=0.7,
            linewidth=1.5,
            label="Mean",
            zorder=3
        )
        ax.legend(frameon=False)

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=10,
        loc="left"
    )

    ax.set_xticks([])
    ax.set_xlabel("")

    plt.show()

def multiple_boxplots(data: pl.DataFrame, columns: List[str], title: str = "Title"):
    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.5), 6))  # auto-scale width

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)

    box_data = [data[str(col)].drop_nulls().to_list() for col in columns]
    
    _ = ax.boxplot(
        box_data,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", zorder=1),
        whiskerprops=dict(color="black", zorder=1),
        capprops=dict(color="black", zorder=1),
        medianprops=dict(color="black", linewidth=2, zorder=2)
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")
    ax.set_xticks(range(1, len(columns) + 1))
    ax.set_xticklabels(columns)
    ax.set_xlabel("")

    plt.tight_layout()
    plt.show()


def multiple_barplot(data: pl.DataFrame, columns: List[str], title: str = "Title", xtick_labels = [], metric = "mean"):
    """
    Plot a bar chart with the mean of each specified numerical column in a Polars DataFrame.

    Parameters
    ----------
    data : pl.DataFrame
        The input DataFrame containing the data.
    columns : List[str]
        List of column names whose means will be plotted.
    title : str, optional
        Title of the plot.
    """
    means = []
    if metric == "mean":
        means = [data[str(col)].drop_nulls().mean() for col in columns]
    elif metric == "median":
        means = [data[str(col)].drop_nulls().median() for col in columns]

    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.5), 6))

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)

    bars = ax.bar(columns, means, color="skyblue", zorder=1)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="grey"
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")
    if len(xtick_labels) > 0:
        ax.set_xticklabels(xtick_labels)
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()

def multiple_kde_plots(data: pl.DataFrame, columns: List[str], title: str = "KDE Plots"):
    assert len(columns) <= 10, "Can only plot up to 10 columns"

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for i, col in enumerate(columns):
        row, col_idx = divmod(i, 5)
        ax = axes[row][col_idx]

        sns.kdeplot(data[col].drop_nulls().to_pandas(), ax=ax, fill=True, alpha=0.3)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # Hide any unused subplots
    for j in range(len(columns), 10):
        row, col_idx = divmod(j, 5)
        axes[row][col_idx].axis("off")

    plt.tight_layout()
    plt.show()

def scatter_plot(data: pl.DataFrame, x: str, y: str, title: str = "Title"):
    fig, ax = plt.subplots(figsize=(10, 6))

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
    
    _ = sns.scatterplot(
        data,
        x=x, y=y
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")
    # ax.set_xticks(range(1, len(columns) + 1))
    # ax.set_xticklabels(columns)
    # ax.set_xlabel("")

    plt.tight_layout()
    plt.show()

def plot_training_loss_history(history, loss = "Loss"):
    plt.plot(history.history['loss'], label=f'Train {loss}')
    plt.plot(history.history['val_loss'], label=f'Val {loss}')
    plt.xlabel('Epoch')
    plt.ylabel(loss)
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.show()


def compute_and_plot_pr_roc_auc(y_true, y_probas):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probas)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probas)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    # Precision-Recall
    ax = axes[1]
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()