import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from helper import create_file_if_not_exists


def plot_storage_reduction(overall_results_df: pd.DataFrame, individual_results_df: pd.DataFrame, output_dir):
    """
    Generate separate seaborn-styled plots showing storage reduction,
    compression efficiency, cluster relationships, and compute tradeoffs.
    Saves each plot as an individual file.
    """

    # Convert percentage column to float if needed
    if overall_results_df['percentage_of_original_size'].dtype == object:
        pct = overall_results_df['percentage_of_original_size'].str.rstrip('%').astype(float)
    else:
        pct = overall_results_df['percentage_of_original_size']

    if individual_results_df['percentage_of_original_size'].dtype == str:
        # print(f"{individual_results_df['percentage_of_original_size'].dtype}")
        ipct = individual_results_df['percentage_of_original_size'].str.rstrip('%').astype(float)
    else:
        ipct = individual_results_df['percentage_of_original_size']

    indices = overall_results_df["columns_decayed"].tolist()
    predicted = overall_results_df["predicted_feature"].tolist()
    original_bits = overall_results_df['original_bits'].tolist()
    final_bits = overall_results_df['bits'].tolist()
    table_data = list(zip(indices, predicted, original_bits, final_bits))

    # Create a table for attribute index
    plt.figure(figsize=(8, 0.5 * len(table_data) + 1))
    plt.title("Attribute Index → Predicted Feature Mapping", pad=20)
    plt.axis("off")

    # Add the table
    table = plt.table(
        cellText=table_data,
        colLabels=["Attribute Index", "Predicted Feature", "Original Bits", "Final Bits"],
        loc="center",
        cellLoc="left"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.4)

    plt.tight_layout()
    create_file_if_not_exists(output_dir / 'attribute_index_mapping.png')
    plt.savefig(output_dir / "attribute_index_mapping.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    sns.set_theme(style="whitegrid", context="talk")

    # Original vs Compressed Size
    x = range(len(overall_results_df))

    plt.figure(figsize=(12, 6))

    # First barplot (Original)
    ax = sns.barplot(
        x=list(x),
        y=overall_results_df["original_size (bytes)"],
        color="skyblue",
        label="Original"
    )

    # Count how many bars exist after the first plot
    n_original = len(ax.patches)

    # Apply hatch to original bars only
    for bar in ax.patches:
        bar.set_hatch("//")

    # Second barplot (Compressed)
    ax2 = sns.barplot(
        x=list(x),
        y=overall_results_df["size (bytes)"],
        color="salmon",
        label="Compressed"
    )

    # New bars are appended to ax.patches, so slice the tail
    new_bars = ax.patches[n_original:]

    # Apply hatch to compressed bars only
    for bar in new_bars:
        bar.set_hatch("--")

    plt.title("Original vs Compressed Size")
    plt.xlabel("Attribute Index")
    plt.ylabel("Bytes")
    plt.xticks(ticks=list(x), labels=list(x))
    plt.legend()
    plt.tight_layout()
    create_file_if_not_exists(output_dir / "original_vs_compressed.png")
    plt.savefig(output_dir / "original_vs_compressed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Breakdown of Compressed Size
    x = range(len(overall_results_df))

    plt.figure(figsize=(12, 6))

    # First barplot (Original)
    ax = sns.barplot(
        x=list(x),
        y=individual_results_df["size (bytes)"],
        color="skyblue",
        label="Compressed"
    )

    # Count how many bars exist after the first plot
    n_original = len(ax.patches)

    # Apply hatch to original bars only
    for bar in ax.patches:
        bar.set_hatch("//")

    # Second barplot (Clustered Models)
    ax2 = sns.barplot(
        x=list(x),
        y=individual_results_df["models_size (bytes)"],
        color="salmon",
        label="Clustered Models"
    )

    # New bars are appended to ax.patches, so slice the tail
    new_bars = ax.patches[n_original:]

    # Apply hatch to compressed bars only
    for bar in new_bars:
        bar.set_hatch("--")

    ax3 = sns.barplot(
        x=list(x),
        y=individual_results_df["outlier_size (bytes)"],
        color="green",
        label="Outliers"
    )

    # New bars are appended to ax.patches, so slice the tail
    new_bars = ax.patches[n_original:]

    # Apply hatch to compressed bars only
    for bar in new_bars:
        bar.set_hatch("++")

    plt.title("Overheads of Compressed Size")
    plt.xlabel("Attribute Index")
    plt.ylabel("Reduced Column Bytes")
    plt.xticks(ticks=list(x), labels=list(x))
    plt.legend()
    plt.tight_layout()
    create_file_if_not_exists(output_dir / "overheads_of_compressed.png")
    plt.savefig(output_dir / "overheads_of_compressed.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Percentage Reduction
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x=x,
        y=pct,
        data=overall_results_df,
        palette="viridis",
        hue=x,
        legend=False
    )

    # Apply hatch patterns
    single_hatch = "//"
    for bar in ax.patches:
        bar.set_hatch(single_hatch)

    plt.title("Compression Efficiency (% of Original Size)")
    plt.xlabel("Attribute Index")
    plt.ylabel("Percentage")
    plt.xticks(ticks=list(x), labels=list(x))
    plt.tight_layout()
    create_file_if_not_exists(output_dir / "percentage_reduction.png")
    plt.savefig(output_dir / "percentage_reduction.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Cumulative Compression
    plt.figure(figsize=(10, 6))

    # Accessible line styles
    line_styles = ["-", "--", "-.", ":"]
    x = range(len(overall_results_df))

    plt.plot(
        x,
        overall_results_df["size (bytes)"].cumsum(),
        label="Compressed",
        linewidth=3,
        linestyle=line_styles[1]
    )
    plt.plot(
        x,
        overall_results_df["original_size (bytes)"].cumsum(),
        label="Original",
        linewidth=3,
        linestyle=line_styles[0]
    )

    plt.title("Cumulative Storage Cost Across Features")
    plt.xlabel("Attribute Index")
    plt.ylabel("Cumulative Bytes")
    plt.xticks(ticks=list(x), labels=list(x))
    plt.legend()
    plt.tight_layout()
    create_file_if_not_exists(output_dir / "cumulative_compression.png")
    plt.savefig(output_dir / "cumulative_compression.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Outliers vs Compression
    plt.figure(figsize=(10, 6))

    # Accessible marker styles
    markers = [
        "o", "s", "D", "^", "v", "<", ">", "P", "X", "*",
        "h", "H", "d", "p", "8"
    ]

    unique_feats = overall_results_df["predicted_feature"].unique()

    # Build a stable, collision‑free mapping
    marker_map = {
        feat: markers[i % len(markers)]
        for i, feat in enumerate(unique_feats)
    }

    sns.scatterplot(
        x="num_outliers",
        y=ipct,
        data=individual_results_df,
        hue="predicted_feature",
        palette="tab20",  # more colors → fewer collisions
        style="predicted_feature",
        markers=marker_map,
        s=150,
        legend="full"
    )

    plt.title("Outliers vs Compression Efficiency")
    plt.xlabel("Number of Outliers")
    plt.ylabel("Percentage of Column Size")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    create_file_if_not_exists(output_dir / "outliers_vs_compression.png")
    plt.savefig(output_dir / "outliers_vs_compression.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Clusters vs Compression
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x="num_clusters",
        y=ipct,
        data=individual_results_df,
        hue="predicted_feature",
        palette="tab20",  # more colors → fewer collisions
        style="predicted_feature",
        markers=marker_map,
        s=150,
        legend="full"
    )

    plt.title("Clusters vs Compression Efficiency")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Percentage of Column Size")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    create_file_if_not_exists(output_dir / "clusters_vs_compression.png")
    plt.savefig(output_dir / "clusters_vs_compression.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_plot(fig, outdir, filename):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main(data_name: str):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import seaborn as sns

    root_dir = Path('.') / 'results' / 'analysis' / 'errors' / f'{data_name}'

    # Load data
    post_df = pd.read_csv(Path('.') / 'results' / 'linear_regression' / f"{data_name}_output_summary.csv")
    lossy_df = pd.read_csv(Path('.') / 'results' / 'compression' / f"{data_name}_error_results.csv")

    # Normalize postdiction
    def normalize_postdiction(df):
        records = []
        for _, row in df.iterrows():
            feature = row["predicted_feature"]

            records.append({
                "method": "postdiction",
                "feature": feature,
                "error_type": "inlier",
                "mean": row["mean_inlier_error"],
                "median": row["median_inlier_error"],
                "min": row["min_inlier_error"],
                "max": row["max_inlier_error"],
                "std": row["std_inlier_error"]
            })

            records.append({
                "method": "postdiction",
                "feature": feature,
                "error_type": "outlier",
                "mean": row["mean_outlier_error"],
                "median": row["median_outlier_error"],
                "min": row["min_outlier_error"],
                "max": row["max_outlier_error"],
                "std": row["std_outlier_error"]
            })
        return pd.DataFrame(records)

    # Normalize lossy compression
    def normalize_lossy(df):
        records = []
        for _, row in df.iterrows():
            feature = row["column"]
            method = row["method"]

            records.append({
                "method": method,
                "feature": feature,
                "error_type": "inlier",
                "mean": row["inliers_mean_error"],
                "median": row["inliers_median_error"],
                "min": row["inliers_min_error"],
                "max": row["inliers_max_error"],
                "std": row["inliers_stdev_error"]
            })

            records.append({
                "method": method,
                "feature": feature,
                "error_type": "outlier",
                "mean": row["outliers_mean_error"],
                "median": row["outliers_median_error"],
                "min": row["outliers_min_error"],
                "max": row["outliers_max_error"],
                "std": row["outliers_stdev_error"]
            })
        return pd.DataFrame(records)

    # Combine and clean
    post_long = normalize_postdiction(post_df)
    lossy_long = normalize_lossy(lossy_df)
    combined = pd.concat([post_long, lossy_long], ignore_index=True)

    numeric_cols = ["mean", "median", "min", "max", "std"]
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors="coerce")
    combined[numeric_cols] = combined[numeric_cols].fillna(0)

    # Melt into long format for boxplots
    long_stats = combined.melt(
        id_vars=["method", "feature", "error_type"],
        value_vars=["mean", "median", "min", "max", "std"],
        var_name="statistic",
        value_name="value"
    )

    # Split into inliers and outliers
    inliers = long_stats[long_stats["error_type"] == "inlier"]
    outliers = long_stats[long_stats["error_type"] == "outlier"]

    # Plot: Inliers
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(18, 8))

    sns.boxplot(
        data=inliers,
        x="feature",
        y="value",
        hue="method",
        palette="Set2"
    )

    # Overlay method-colored points
    sns.stripplot(
        data=inliers,
        x="feature",
        y="value",
        hue="method",
        dodge=True,
        palette="Set2",
        size=6,
        alpha=0.8,
        jitter=False,
        marker="o"
    )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(1.00, color="red", linestyle="--", linewidth=2)
    plt.title("Per-Column Error Statistics — Inliers")
    plt.ylabel("Error Magnitude")
    plt.xlabel("Feature")
    plt.tight_layout()
    save_plot(plt, root_dir, 'box-inliers.png')

    # Plot: Outliers
    plt.figure(figsize=(18, 8))

    sns.boxplot(
        data=outliers,
        x="feature",
        y="value",
        hue="method",
        palette="Set2"
    )

    sns.stripplot(
        data=outliers,
        x="feature",
        y="value",
        hue="method",
        dodge=True,
        palette="Set2",
        size=6,
        alpha=0.8,
        jitter=False,
        marker="o"
    )

    plt.xticks(rotation=45, ha="right")
    plt.axhline(1.00, color="red", linestyle="--", linewidth=2)
    plt.title("Per-Column Error Statistics — Outliers")
    plt.ylabel("Error Magnitude")
    plt.xlabel("Feature")
    plt.tight_layout()
    save_plot(plt, root_dir, 'box-outliers.png')

    def plot_cdf(df, title, file_title):
        plt.figure(figsize=(14, 8))

        methods = df["method"].unique()
        colors = sns.color_palette("Set2", n_colors=len(methods))

        for method, color in zip(methods, colors):
            subset = df[df["method"] == method]["value"].values
            subset = np.sort(subset)

            # Compute CDF
            y = np.linspace(0, 1, len(subset))

            plt.plot(subset, y, label=method, linewidth=3, color=color)

        # Tolerance line at 1.00
        plt.axvline(1.00, color="red", linestyle="--", linewidth=2)

        plt.xlabel("Relative Error (0–1 scale)")
        plt.ylabel("CDF")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(title="Method")
        plt.tight_layout()
        save_plot(plt, root_dir, f'cdf-{file_title}.png')

    # CDF for Inliers
    plot_cdf(
        df=inliers[inliers["statistic"].isin(["mean", "median", "min", "max", "std"])],
        title="CDF of Per-Column Error Statistics — Inliers",
        file_title='inliers'
    )

    # CDF for Outliers
    plot_cdf(
        df=outliers[outliers["statistic"].isin(["mean", "median", "min", "max", "std"])],
        title="CDF of Per-Column Error Statistics — Outliers",
        file_title='outliers'
    )


if __name__ == "__main__":
    for dataset_name in ['health', 'river', 'air']:
        main(dataset_name)
