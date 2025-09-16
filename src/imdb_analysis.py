"""Download and analyze the IMDb 1000 sample dataset.

The script downloads a zipped archive of the `imdb_1000` dataset from the
DAT8 teaching repository, extracts the CSV file, and reproduces the insights
highlighted during exploratory analysis:

* Westerns' unusually strong IMDb ratings despite the genre's tiny presence
  in the sample.
* History films' short runtimes paired with high scores.
* The positive runtime-to-rating relationship across the catalogue.
* Actor "efficiency" – performers whose limited appearances receive
  consistently high marks.
* Appreciation for epic-length films.

Running the script produces charts in ``charts/`` and a Markdown report in
``reports/`` describing the findings with context and supporting numbers.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"

DATA_URL = "https://github.com/justmarkham/DAT8/archive/refs/heads/master.zip"
ZIP_PATH = RAW_DIR / "dat8_master.zip"
CSV_MEMBER = "DAT8-master/data/imdb_1000.csv"
CSV_PATH = PROCESSED_DIR / "imdb_1000.csv"

INDIAN_CORE_TITLES = {
    "3 Idiots",
    "Taare Zameen Par",
    "Munna Bhai M.B.B.S.",
    "Rang De Basanti",
    "Swades",
    "Dilwale Dulhania Le Jayenge",
    "Dil Chahta Hai",
    "Barfi!",
    "Lagaan: Once Upon a Time in India",
    "My Name Is Khan",
}

INDIAN_COPRO_TITLES = {
    "Life of Pi",
    "Slumdog Millionaire",
}

KNOWN_INDIAN_ACTORS = {
    "Aamir Khan",
    "Abhishek Bachchan",
    "Adil Hussain",
    "Aishwarya Rai",
    "Akshaye Khanna",
    "Alia Bhatt",
    "Amrish Puri",
    "Anil Kapoor",
    "Arjun Kapoor",
    "Arshad Warsi",
    "Darsheel Safary",
    "Dev Patel",
    "Dhanush",
    "Dilip Kumar",
    "Farhan Akhtar",
    "Freida Pinto",
    "Gayatri Joshi",
    "Gracy Singh",
    "Hema Malini",
    "Ileana",
    "Irrfan",
    "Irrfan Khan",
    "Kajol",
    "Kangana Ranaut",
    "Kareena Kapoor",
    "Katrina Kaif",
    "Kishori Balal",
    "Konkona Sen Sharma",
    "Madhavan",
    "Madhuri Dixit",
    "Mammootty",
    "Manoj Bajpayee",
    "Mithun Chakraborty",
    "Mohanlal",
    "Mona Singh",
    "Naseeruddin Shah",
    "Nayanthara",
    "Nawazuddin Siddiqui",
    "Pankaj Tripathi",
    "Parineeti Chopra",
    "Priyanka Chopra",
    "Radhika Apte",
    "Rajesh Khanna",
    "Rajinikanth",
    "Rajkummar Rao",
    "Ranbir Kapoor",
    "Ranveer Singh",
    "Rani Mukerji",
    "Saif Ali Khan",
    "Salman Khan",
    "Sanjay Dutt",
    "Saurabh Shukla",
    "Shah Rukh Khan",
    "Sheetal Menon",
    "Siddharth",
    "Soha Ali Khan",
    "Sunil Dutt",
    "Suriya",
    "Suraj Sharma",
    "Taapsee Pannu",
    "Tanay Chheda",
    "Vidya Balan",
    "Vijay Sethupathi",
}


def download_dataset(url: str = DATA_URL, destination: Path = ZIP_PATH) -> Path:
    """Download the dataset archive if it is not already cached."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Reusing cached archive at {destination}")
        return destination

    print(f"Downloading dataset archive from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"Saved archive to {destination}")
    return destination


def extract_csv(
    zip_path: Path = ZIP_PATH, member_name: str = CSV_MEMBER, output_path: Path = CSV_PATH
) -> Path:
    """Extract the IMDb CSV from the downloaded archive."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Reusing extracted CSV at {output_path}")
        return output_path

    with zipfile.ZipFile(zip_path) as archive:
        if member_name not in archive.namelist():
            raise FileNotFoundError(
                f"Expected member '{member_name}' not found in archive {zip_path}"
            )
        with archive.open(member_name) as source, output_path.open("wb") as target:
            target.write(source.read())
    print(f"Extracted IMDb CSV to {output_path}")
    return output_path


def load_dataset(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the IMDb sample dataset into a DataFrame."""

    df = pd.read_csv(csv_path)
    df["actors_list"] = df["actors_list"].apply(ast.literal_eval)
    df["genre"] = df["genre"].astype("category")
    return df


def compute_genre_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated statistics by genre."""

    genre_stats = (
        df.groupby("genre", observed=True)
        .agg(
            mean_rating=("star_rating", "mean"),
            count=("title", "count"),
            avg_duration=("duration", "mean"),
        )
        .sort_values("mean_rating", ascending=False)
    )
    return genre_stats


def compute_actor_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated statistics by actor."""

    exploded = (
        df[["title", "star_rating", "actors_list"]]
        .explode("actors_list")
        .rename(columns={"actors_list": "actor"})
    )
    actor_stats = (
        exploded.groupby("actor")
        .agg(film_count=("title", "count"), mean_rating=("star_rating", "mean"))
        .sort_values("mean_rating", ascending=False)
    )
    return actor_stats


def _count_indian_actors(actors: list[str]) -> int:
    return sum(actor in KNOWN_INDIAN_ACTORS for actor in actors)


def annotate_indian_origins(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` labelled with Indian origin metadata."""

    labelled = df.copy()
    labelled["indian_actor_count"] = labelled["actors_list"].apply(_count_indian_actors)

    def classify(row: pd.Series) -> str:
        title = row["title"]
        if title in INDIAN_CORE_TITLES:
            return "Indian core"
        if title in INDIAN_COPRO_TITLES:
            return "Indian diaspora co-production"
        if row["indian_actor_count"] >= 2:
            return "Indian (inferred)"
        return "Other"

    labelled["origin_detail"] = labelled.apply(classify, axis=1)
    detail_to_group = {
        "Indian core": "Indian",
        "Indian diaspora co-production": "Indian",
        "Indian (inferred)": "Indian",
    }
    labelled["origin_group"] = (
        labelled["origin_detail"].map(detail_to_group).fillna("Other")
    )
    return labelled


def summarize_origin_groups(
    labelled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary tables for Indian vs. other origin groupings."""

    def _summarize(group_col: str) -> pd.DataFrame:
        summary = (
            labelled.groupby(group_col, observed=True)
            .agg(
                film_count=("title", "count"),
                mean_rating=("star_rating", "mean"),
                median_rating=("star_rating", "median"),
                min_rating=("star_rating", "min"),
                max_rating=("star_rating", "max"),
            )
            .sort_values("mean_rating", ascending=False)
        )
        return summary

    return _summarize("origin_group"), _summarize("origin_detail")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved chart to {path}")


def generate_genre_rating_chart(genre_stats: pd.DataFrame, output_path: Path) -> None:
    """Plot average IMDb rating by genre with observation counts."""

    data = genre_stats.reset_index()
    order = data.sort_values("mean_rating", ascending=True)

    colors = sns.color_palette("viridis", len(order))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(order["genre"], order["mean_rating"], color=colors)
    ax.set_xlabel("Average IMDb rating")
    ax.set_ylabel("")
    ax.set_title("Average rating by genre (IMDb Top 1000 subset)")

    for bar, count in zip(bars, order["count"]):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(count)}",
            va="center",
            fontsize=9,
            color="dimgray",
        )

    _save_figure(fig, output_path)


def generate_genre_runtime_chart(genre_stats: pd.DataFrame, output_path: Path) -> None:
    """Plot genre-level runtime vs. rating with emphasis on niche genres."""

    data = genre_stats.reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        data["avg_duration"],
        data["mean_rating"],
        s=40 + data["count"] * 6,
        c=data["count"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_xlabel("Average runtime (minutes)")
    ax.set_ylabel("Average IMDb rating")
    ax.set_title("Runtime vs. rating by genre")

    for _, row in data.iterrows():
        if row["genre"] in {"History", "Western"}:
            ax.annotate(
                row["genre"],
                (row["avg_duration"], row["mean_rating"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="darkred" if row["genre"] == "History" else "darkblue",
            )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of films in genre")

    _save_figure(fig, output_path)


def generate_runtime_rating_chart(df: pd.DataFrame, output_path: Path) -> float:
    """Plot runtime vs. rating for individual films and return the correlation."""

    correlation = float(df["duration"].corr(df["star_rating"]))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.regplot(
        data=df,
        x="duration",
        y="star_rating",
        scatter_kws={"alpha": 0.45},
        line_kws={"color": "#1f4e79"},
        ax=ax,
    )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("IMDb rating")
    ax.set_title("Longer films trend toward higher ratings")
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    _save_figure(fig, output_path)
    return correlation


def generate_indian_vs_other_chart(
    labelled: pd.DataFrame, summary: pd.DataFrame, output_path: Path
) -> None:
    """Visualise IMDb ratings for Indian versus non-Indian titles."""

    relevant = labelled[labelled["origin_group"].isin({"Indian", "Other"})]
    if relevant.empty:
        raise ValueError("No records available for Indian/origin comparison")

    order = ["Indian", "Other"]
    palette = {"Indian": "#4527a0", "Other": "#455a64"}

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.boxplot(
        data=relevant,
        x="origin_group",
        y="star_rating",
        order=order,
        showcaps=False,
        boxprops={"facecolor": "#d1c4e9", "alpha": 0.55},
        whiskerprops={"color": "#5e35b1"},
        medianprops={"color": "#5e35b1", "linewidth": 2},
        width=0.45,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=relevant,
        x="origin_group",
        y="star_rating",
        order=order,
        hue="origin_group",
        jitter=0.12,
        dodge=False,
        alpha=0.85,
        size=6,
        palette=palette,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_xlabel("")
    ax.set_ylabel("IMDb rating")
    ax.set_title("Indian-led films outperform the broader sample")

    for idx, group in enumerate(order):
        if group in summary.index:
            stats = summary.loc[group]
            ax.text(
                idx,
                stats["mean_rating"] + 0.03,
                f"mean {stats['mean_rating']:.2f}\n(n={int(stats['film_count'])})",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color=palette.get(group, "black"),
            )

    if all(group in summary.index for group in order):
        diff = summary.loc["Indian", "mean_rating"] - summary.loc["Other", "mean_rating"]
        ax.text(
            0.5,
            0.05,
            f"Mean difference: {diff:+.2f}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    _save_figure(fig, output_path)


def generate_actor_efficiency_chart(
    actor_stats: pd.DataFrame, output_path: Path, min_films: int = 3, top_n: int = 15
) -> pd.DataFrame:
    """Plot average rating for actors with at least ``min_films`` titles."""

    filtered = actor_stats.query("film_count >= @min_films").copy()
    if filtered.empty:
        raise ValueError("No actors meet the minimum film threshold")

    ordered = filtered.sort_values(["mean_rating", "film_count"], ascending=[False, False])
    top = ordered.head(top_n).sort_values("mean_rating")
    top = top.reset_index().rename(columns={"index": "actor"})
    top["highlight"] = top["actor"] == "Mark Hamill"

    colors = ["#e74c3c" if is_mark else "#90a4ae" for is_mark in top["highlight"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["actor"], top["mean_rating"], color=colors)
    ax.set_xlabel("Average IMDb rating")
    ax.set_ylabel("")
    ax.set_title("Actors with consistently high-rated performances")

    for bar, (_, row) in zip(bars, top.iterrows()):
        text_x = bar.get_width() - 0.05
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean_rating']:.2f} (n={int(row['film_count'])})",
            va="center",
            ha="right",
            color="white" if row["highlight"] else "black",
            fontsize=9,
        )

    _save_figure(fig, output_path)
    return top.sort_values("mean_rating", ascending=False)


def generate_epic_runtime_chart(df: pd.DataFrame, output_path: Path, top_n: int = 8) -> pd.DataFrame:
    """Plot the longest films in the dataset and return the subset used."""

    longest = df.nlargest(top_n, "duration").copy()
    ordered = longest.sort_values("duration")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=ordered,
        x="duration",
        y="title",
        hue="star_rating",
        palette="magma",
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("")
    ax.set_title("Even the longest films maintain strong ratings")
    ax.get_legend().set_title("IMDb rating")

    for index, row in enumerate(ordered.itertuples()):
        ax.text(
            row.duration + 1,
            index,
            f"{row.star_rating:.1f}",
            va="center",
            color="black",
        )

    _save_figure(fig, output_path)
    return ordered.sort_values("duration", ascending=False)


def create_report(
    df: pd.DataFrame,
    genre_stats: pd.DataFrame,
    actor_stats: pd.DataFrame,
    correlation: float,
    top_actor_slice: pd.DataFrame,
    long_films: pd.DataFrame,
    origin_summary: pd.DataFrame,
    origin_detail: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a Markdown report summarizing the analysis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    western = genre_stats.loc["Western"] if "Western" in genre_stats.index else None
    history = genre_stats.loc["History"] if "History" in genre_stats.index else None

    efficiency_leader = top_actor_slice.sort_values("mean_rating", ascending=False).head(1)

    report_lines = [
        "# IMDb 1000 sample analysis",
        "",
        "## Dataset",
        f"* Source archive: [{DATA_URL}]({DATA_URL})",
        f"* Records analysed: {len(df)} films",
        "",
        "## Key findings",
    ]

    if western is not None:
        report_lines.append(
            f"- **Westerns quietly dominate ratings.** The genre averages {western['mean_rating']:.2f}"
            f" across just {int(western['count'])} titles, far outpacing more saturated categories."
        )
    if history is not None:
        report_lines.append(
            f"- **History titles deliver impact in minimal time.** The lone entry runs {history['avg_duration']:.0f}"
            f" minutes yet still posts a {history['mean_rating']:.1f} rating, signalling an underexplored format."
        )
    report_lines.append(
        f"- **Runtime and reception move together.** Across the sample, runtime and rating show a Pearson"
        f" correlation of {correlation:.2f}, indicating longer stories earn modestly higher scores."
    )
    if not top_actor_slice.empty:
        leader = efficiency_leader.iloc[0]
        report_lines.append(
            f"- **Actor efficiency reveals hidden standouts.** {leader['actor']} averages {leader['mean_rating']:.2f}"
            f" from {int(leader['film_count'])} appearances, the strongest record among performers with at least"
            f" three films."
        )
    if not long_films.empty:
        top_long = long_films.head(1).iloc[0]
        report_lines.append(
            f"- **Epic runtimes still resonate.** The longest film in the set, {top_long.title}"
            f" ({int(top_long.duration)} minutes), holds an IMDb rating of {top_long.star_rating:.1f}."
        )
    if {"Indian", "Other"}.issubset(origin_summary.index):
        indian_stats = origin_summary.loc["Indian"]
        other_stats = origin_summary.loc["Other"]
        diff = indian_stats["mean_rating"] - other_stats["mean_rating"]
        direction = "higher" if diff >= 0 else "lower"
        report_lines.append(
            "- **Indian-led films defy the low-rating hypothesis.** The curated Indian sample posts an average "
            f"rating of {indian_stats['mean_rating']:.2f} across {int(indian_stats['film_count'])} titles, "
            f"{abs(diff):.2f} points {direction} than the {other_stats['mean_rating']:.2f} mean for other regions."
        )

    report_lines.extend(
        [
            "",
            "## Genre overview",
            "", "| Genre | Avg rating | Films | Avg runtime (min) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for genre, row in genre_stats.sort_values("mean_rating", ascending=False).iterrows():
        report_lines.append(
            f"| {genre} | {row['mean_rating']:.2f} | {int(row['count'])} | {row['avg_duration']:.0f} |"
        )

    report_lines.extend(
        [
            "",
            "## Actor efficiency (≥3 films)",
            "", "| Actor | Films | Avg rating |",
            "| --- | ---: | ---: |",
        ]
    )

    for _, row in top_actor_slice.iterrows():
        report_lines.append(
            f"| {row['actor']} | {int(row['film_count'])} | {row['mean_rating']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Longest films",
            "", "| Title | Runtime (min) | IMDb rating |",
            "| --- | ---: | ---: |",
        ]
    )

    for _, row in long_films.iterrows():
        report_lines.append(
            f"| {row['title']} | {int(row['duration'])} | {row['star_rating']:.1f} |"
        )

    report_lines.extend(
        [
            "",
            "## Indian vs. other ratings",
            "",
            "| Origin | Films | Avg rating | Median | Min | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for origin, row in origin_summary.iterrows():
        report_lines.append(
            f"| {origin} | {int(row['film_count'])} | {row['mean_rating']:.2f} | {row['median_rating']:.2f} | "
            f"{row['min_rating']:.1f} | {row['max_rating']:.1f} |"
        )

    detail_slice = origin_detail.drop(index=["Other"], errors="ignore")
    if not detail_slice.empty:
        report_lines.extend(
            [
                "",
                "### Breakdown of Indian-linked titles",
                "",
                "| Category | Films | Avg rating | Median |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for origin, row in detail_slice.iterrows():
            report_lines.append(
                f"| {origin} | {int(row['film_count'])} | {row['mean_rating']:.2f} | {row['median_rating']:.2f} |"
            )

    report_lines.append("")

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report to {output_path}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    archive_path = download_dataset()
    csv_path = extract_csv(archive_path)
    df = load_dataset(csv_path)

    genre_stats = compute_genre_stats(df)
    actor_stats = compute_actor_stats(df)
    labelled = annotate_indian_origins(df)
    origin_summary, origin_detail = summarize_origin_groups(labelled)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    generate_genre_rating_chart(genre_stats, CHARTS_DIR / "genre_ratings.png")
    generate_genre_runtime_chart(genre_stats, CHARTS_DIR / "genre_runtime_vs_rating.png")
    correlation = generate_runtime_rating_chart(df, CHARTS_DIR / "runtime_vs_rating.png")
    actor_slice = generate_actor_efficiency_chart(
        actor_stats, CHARTS_DIR / "actor_efficiency.png", min_films=3, top_n=15
    )
    long_films = generate_epic_runtime_chart(df, CHARTS_DIR / "epic_runtime_ratings.png", top_n=8)
    generate_indian_vs_other_chart(
        labelled, origin_summary, CHARTS_DIR / "indian_vs_other_ratings.png"
    )

    create_report(
        df=df,
        genre_stats=genre_stats,
        actor_stats=actor_stats,
        correlation=correlation,
        top_actor_slice=actor_slice,
        long_films=long_films,
        origin_summary=origin_summary,
        origin_detail=origin_detail,
        output_path=REPORTS_DIR / "insights.md",
    )

    print("Analysis complete. Charts available in the 'charts' directory.")


if __name__ == "__main__":
    main()

"""Download and analyze the IMDb 1000 sample dataset.

The script downloads a zipped archive of the `imdb_1000` dataset from the
DAT8 teaching repository, extracts the CSV file, and reproduces the insights
highlighted during exploratory analysis:

* Westerns' unusually strong IMDb ratings despite the genre's tiny presence
  in the sample.
* History films' short runtimes paired with high scores.
* The positive runtime-to-rating relationship across the catalogue.
* Actor "efficiency" – performers whose limited appearances receive
  consistently high marks.
* Appreciation for epic-length films.

Running the script produces charts in ``charts/`` and a Markdown report in
``reports/`` describing the findings with context and supporting numbers.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"

DATA_URL = "https://github.com/justmarkham/DAT8/archive/refs/heads/master.zip"
ZIP_PATH = RAW_DIR / "dat8_master.zip"
CSV_MEMBER = "DAT8-master/data/imdb_1000.csv"
CSV_PATH = PROCESSED_DIR / "imdb_1000.csv"


def download_dataset(url: str = DATA_URL, destination: Path = ZIP_PATH) -> Path:
    """Download the dataset archive if it is not already cached."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Reusing cached archive at {destination}")
        return destination

    print(f"Downloading dataset archive from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    print(f"Saved archive to {destination}")
    return destination


def extract_csv(
    zip_path: Path = ZIP_PATH, member_name: str = CSV_MEMBER, output_path: Path = CSV_PATH
) -> Path:
    """Extract the IMDb CSV from the downloaded archive."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Reusing extracted CSV at {output_path}")
        return output_path

    with zipfile.ZipFile(zip_path) as archive:
        if member_name not in archive.namelist():
            raise FileNotFoundError(
                f"Expected member '{member_name}' not found in archive {zip_path}"
            )
        with archive.open(member_name) as source, output_path.open("wb") as target:
            target.write(source.read())
    print(f"Extracted IMDb CSV to {output_path}")
    return output_path


def load_dataset(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the IMDb sample dataset into a DataFrame."""

    df = pd.read_csv(csv_path)
    df["actors_list"] = df["actors_list"].apply(ast.literal_eval)
    df["genre"] = df["genre"].astype("category")
    return df


def compute_genre_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated statistics by genre."""

    genre_stats = (
        df.groupby("genre", observed=True)
        .agg(
            mean_rating=("star_rating", "mean"),
            count=("title", "count"),
            avg_duration=("duration", "mean"),
        )
        .sort_values("mean_rating", ascending=False)
    )
    return genre_stats


def compute_actor_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated statistics by actor."""

    exploded = (
        df[["title", "star_rating", "actors_list"]]
        .explode("actors_list")
        .rename(columns={"actors_list": "actor"})
    )
    actor_stats = (
        exploded.groupby("actor")
        .agg(film_count=("title", "count"), mean_rating=("star_rating", "mean"))
        .sort_values("mean_rating", ascending=False)
    )
    return actor_stats


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved chart to {path}")


def generate_genre_rating_chart(genre_stats: pd.DataFrame, output_path: Path) -> None:
    """Plot average IMDb rating by genre with observation counts."""

    data = genre_stats.reset_index()
    order = data.sort_values("mean_rating", ascending=True)

    colors = sns.color_palette("viridis", len(order))

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(order["genre"], order["mean_rating"], color=colors)
    ax.set_xlabel("Average IMDb rating")
    ax.set_ylabel("")
    ax.set_title("Average rating by genre (IMDb Top 1000 subset)")

    for bar, count in zip(bars, order["count"]):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(count)}",
            va="center",
            fontsize=9,
            color="dimgray",
        )

    _save_figure(fig, output_path)


def generate_genre_runtime_chart(genre_stats: pd.DataFrame, output_path: Path) -> None:
    """Plot genre-level runtime vs. rating with emphasis on niche genres."""

    data = genre_stats.reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        data["avg_duration"],
        data["mean_rating"],
        s=40 + data["count"] * 6,
        c=data["count"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="black",
    )

    ax.set_xlabel("Average runtime (minutes)")
    ax.set_ylabel("Average IMDb rating")
    ax.set_title("Runtime vs. rating by genre")

    for _, row in data.iterrows():
        if row["genre"] in {"History", "Western"}:
            ax.annotate(
                row["genre"],
                (row["avg_duration"], row["mean_rating"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="darkred" if row["genre"] == "History" else "darkblue",
            )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of films in genre")

    _save_figure(fig, output_path)


def generate_runtime_rating_chart(df: pd.DataFrame, output_path: Path) -> float:
    """Plot runtime vs. rating for individual films and return the correlation."""

    correlation = float(df["duration"].corr(df["star_rating"]))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.regplot(
        data=df,
        x="duration",
        y="star_rating",
        scatter_kws={"alpha": 0.45},
        line_kws={"color": "#1f4e79"},
        ax=ax,
    )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("IMDb rating")
    ax.set_title("Longer films trend toward higher ratings")
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {correlation:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    _save_figure(fig, output_path)
    return correlation


def generate_actor_efficiency_chart(
    actor_stats: pd.DataFrame, output_path: Path, min_films: int = 3, top_n: int = 15
) -> pd.DataFrame:
    """Plot average rating for actors with at least ``min_films`` titles."""

    filtered = actor_stats.query("film_count >= @min_films").copy()
    if filtered.empty:
        raise ValueError("No actors meet the minimum film threshold")

    ordered = filtered.sort_values(["mean_rating", "film_count"], ascending=[False, False])
    top = ordered.head(top_n).sort_values("mean_rating")
    top = top.reset_index().rename(columns={"index": "actor"})
    top["highlight"] = top["actor"] == "Mark Hamill"

    colors = ["#e74c3c" if is_mark else "#90a4ae" for is_mark in top["highlight"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["actor"], top["mean_rating"], color=colors)
    ax.set_xlabel("Average IMDb rating")
    ax.set_ylabel("")
    ax.set_title("Actors with consistently high-rated performances")

    for bar, (_, row) in zip(bars, top.iterrows()):
        text_x = bar.get_width() - 0.05
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean_rating']:.2f} (n={int(row['film_count'])})",
            va="center",
            ha="right",
            color="white" if row["highlight"] else "black",
            fontsize=9,
        )

    _save_figure(fig, output_path)
    return top.sort_values("mean_rating", ascending=False)


def generate_epic_runtime_chart(df: pd.DataFrame, output_path: Path, top_n: int = 8) -> pd.DataFrame:
    """Plot the longest films in the dataset and return the subset used."""

    longest = df.nlargest(top_n, "duration").copy()
    ordered = longest.sort_values("duration")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=ordered,
        x="duration",
        y="title",
        hue="star_rating",
        palette="magma",
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("")
    ax.set_title("Even the longest films maintain strong ratings")
    ax.get_legend().set_title("IMDb rating")

    for index, row in enumerate(ordered.itertuples()):
        ax.text(
            row.duration + 1,
            index,
            f"{row.star_rating:.1f}",
            va="center",
            color="black",
        )

    _save_figure(fig, output_path)
    return ordered.sort_values("duration", ascending=False)


def create_report(
    df: pd.DataFrame,
    genre_stats: pd.DataFrame,
    actor_stats: pd.DataFrame,
    correlation: float,
    top_actor_slice: pd.DataFrame,
    long_films: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a Markdown report summarizing the analysis."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    western = genre_stats.loc["Western"] if "Western" in genre_stats.index else None
    history = genre_stats.loc["History"] if "History" in genre_stats.index else None

    efficiency_leader = top_actor_slice.sort_values("mean_rating", ascending=False).head(1)

    report_lines = [
        "# IMDb 1000 sample analysis",
        "",
        "## Dataset",
        f"* Source archive: [{DATA_URL}]({DATA_URL})",
        f"* Records analysed: {len(df)} films",
        "",
        "## Key findings",
    ]

    if western is not None:
        report_lines.append(
            f"- **Westerns quietly dominate ratings.** The genre averages {western['mean_rating']:.2f}"
            f" across just {int(western['count'])} titles, far outpacing more saturated categories."
        )
    if history is not None:
        report_lines.append(
            f"- **History titles deliver impact in minimal time.** The lone entry runs {history['avg_duration']:.0f}"
            f" minutes yet still posts a {history['mean_rating']:.1f} rating, signalling an underexplored format."
        )
    report_lines.append(
        f"- **Runtime and reception move together.** Across the sample, runtime and rating show a Pearson"
        f" correlation of {correlation:.2f}, indicating longer stories earn modestly higher scores."
    )
    if not top_actor_slice.empty:
        leader = efficiency_leader.iloc[0]
        report_lines.append(
            f"- **Actor efficiency reveals hidden standouts.** {leader['actor']} averages {leader['mean_rating']:.2f}"
            f" from {int(leader['film_count'])} appearances, the strongest record among performers with at least"
            f" three films."
        )
    if not long_films.empty:
        top_long = long_films.head(1).iloc[0]
        report_lines.append(
            f"- **Epic runtimes still resonate.** The longest film in the set, {top_long.title}"
            f" ({int(top_long.duration)} minutes), holds an IMDb rating of {top_long.star_rating:.1f}."
        )

    report_lines.extend(
        [
            "",
            "## Genre overview",
            "", "| Genre | Avg rating | Films | Avg runtime (min) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )

    for genre, row in genre_stats.sort_values("mean_rating", ascending=False).iterrows():
        report_lines.append(
            f"| {genre} | {row['mean_rating']:.2f} | {int(row['count'])} | {row['avg_duration']:.0f} |"
        )

    report_lines.extend(
        [
            "",
            "## Actor efficiency (≥3 films)",
            "", "| Actor | Films | Avg rating |",
            "| --- | ---: | ---: |",
        ]
    )

    for _, row in top_actor_slice.iterrows():
        report_lines.append(
            f"| {row['actor']} | {int(row['film_count'])} | {row['mean_rating']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Longest films",
            "", "| Title | Runtime (min) | IMDb rating |",
            "| --- | ---: | ---: |",
        ]
    )

    for _, row in long_films.iterrows():
        report_lines.append(
            f"| {row['title']} | {int(row['duration'])} | {row['star_rating']:.1f} |"
        )

    report_lines.append("")

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report to {output_path}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    archive_path = download_dataset()
    csv_path = extract_csv(archive_path)
    df = load_dataset(csv_path)

    genre_stats = compute_genre_stats(df)
    actor_stats = compute_actor_stats(df)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    generate_genre_rating_chart(genre_stats, CHARTS_DIR / "genre_ratings.png")
    generate_genre_runtime_chart(genre_stats, CHARTS_DIR / "genre_runtime_vs_rating.png")
    correlation = generate_runtime_rating_chart(df, CHARTS_DIR / "runtime_vs_rating.png")
    actor_slice = generate_actor_efficiency_chart(
        actor_stats, CHARTS_DIR / "actor_efficiency.png", min_films=3, top_n=15
    )
    long_films = generate_epic_runtime_chart(df, CHARTS_DIR / "epic_runtime_ratings.png", top_n=8)

    create_report(
        df=df,
        genre_stats=genre_stats,
        actor_stats=actor_stats,
        correlation=correlation,
        top_actor_slice=actor_slice,
        long_films=long_films,
        output_path=REPORTS_DIR / "insights.md",
    )

    print("Analysis complete. Charts available in the 'charts' directory.")


if __name__ == "__main__":
    main()
