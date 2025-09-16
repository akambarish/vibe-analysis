# IMDb vibe analysis

This repository automates the exploratory analysis of the open `imdb_1000`
sample (1,000 top-rated titles) that approximates prior manual investigation.
The script downloads the dataset from the [DAT8 teaching
repository](https://github.com/justmarkham/DAT8), extracts the CSV from the zip
archive, computes the same niche insights highlighted earlier, and generates a
set of publication-ready visuals alongside a Markdown summary.

## Requirements

- Python 3.11+
- The packages listed in [`requirements.txt`](requirements.txt)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the analysis from the project root:

```bash
python src/imdb_analysis.py
```

The command will:

1. Download and cache the zipped dataset under `data/raw/`.
2. Extract `imdb_1000.csv` into `data/processed/`.
3. Generate charts in the [`charts/`](charts) directory capturing:
   - Genre-level rating dynamics (including Western and History outliers).
   - The runtime vs. rating relationship.
   - Actor efficiency for performers appearing in three or more titles.
   - Ratings for the longest films in the sample.
4. Write a narrative summary to [`reports/insights.md`](reports/insights.md).

If the archive already exists locally, the script reuses it rather than
re-downloading.

## Outputs

After running the script you will find:

- `charts/genre_ratings.png`
- `charts/genre_runtime_vs_rating.png`
- `charts/runtime_vs_rating.png`
- `charts/actor_efficiency.png`
- `charts/epic_runtime_ratings.png`
- `reports/insights.md`

Each asset captures one of the exploratory insights discovered from the
`imdb_1000` sample and can be used directly in downstream reporting.
