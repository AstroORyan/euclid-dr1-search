"""Benchmark an AnomalyMatch prediction run against a catalogue of known anomalies.

Reads a `predictions.db` produced by a benchmarking run, ranks every source by
anomaly score, and measures how many of the known anomalies are recovered as a
function of how far down the ranked list you look.

Example
-------
    python benchmark_analysis.py --known-csv known_lenses_in_sample.csv --run run1
"""

import argparse
import random
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = "/media/team_workspaces/AnomalyMatch-IDR1-Search/benchmarking_tests/results"
RUN_CHOICES = ("run1", "run2", "run3", "run4", "test_run", "test-run")
SCORE_INDEX_NAME = "idx_score_covering"
DEFAULT_TOP_N = 5_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--known-csv",
        required=True,
        type=Path,
        help="CSV of known anomalies (e.g. known_lenses_in_sample.csv).",
    )
    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Run to analyse: either a name appended to the results root "
            f"({', '.join(RUN_CHOICES)}), or a full path to a run directory."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top-scoring sources to extract (default: {DEFAULT_TOP_N:_}).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(RESULTS_ROOT),
        help="Root directory holding the run directories.",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Column in the known-anomaly CSV holding the source IDs (default: id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write outputs. Defaults to a timestamp_seed directory next to this script.",
    )
    return parser.parse_args(argv)


def resolve_run_dir(results_root, run):
    """Locate the run directory.

    `run` is either a plain run name resolved against `results_root` (tolerating
    test_run vs test-run naming), or a full/relative path to a run directory.
    """
    as_path = Path(run).expanduser()
    if as_path.is_dir():
        return as_path
    if as_path.is_absolute() or len(as_path.parts) > 1:
        raise SystemExit(f"Run directory does not exist: {as_path}")

    candidates = [run, run.replace("_", "-"), run.replace("-", "_")]
    for name in dict.fromkeys(candidates):
        path = results_root / name
        if path.is_dir():
            return path

    available = sorted(p.name for p in results_root.iterdir() if p.is_dir()) if results_root.is_dir() else []
    raise SystemExit(
        f"No run directory for '{run}' under {results_root}.\n"
        f"Available: {', '.join(available) if available else '(results root not found)'}"
    )


@contextmanager
def sqlite_connection(db_path, read_only):
    """Open the database and guarantee it is closed again.

    Writable connections checkpoint the write-ahead log on the way out so the
    `-wal`/`-shm` sidecar files are not left behind next to the database.
    """
    uri = f"file:{db_path}?mode=ro" if read_only else f"file:{db_path}"
    con = sqlite3.connect(uri, uri=True)
    try:
        con.execute("PRAGMA cache_size = -2000000")
        con.execute("PRAGMA temp_store = MEMORY")
        yield con
    finally:
        try:
            if not read_only:
                con.commit()
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()


def ensure_score_index(db_path):
    """Create the covering (score DESC, filename) index if it is not already there."""
    with sqlite_connection(db_path, read_only=False) as con:
        existing = {row[1] for row in con.execute("PRAGMA index_list('results')")}
        if SCORE_INDEX_NAME in existing:
            print(f"Index '{SCORE_INDEX_NAME}' already present.")
        else:
            print(f"Index '{SCORE_INDEX_NAME}' missing - building it (this can take a while)...")
            start = time.perf_counter()
            con.execute(f"CREATE INDEX IF NOT EXISTS {SCORE_INDEX_NAME} ON results (score DESC, filename)")
            con.commit()
            print(f"  built in {time.perf_counter() - start:.1f} s")
    con.close()


def load_top_sources(db_path, top_n):
    """Return the top-N sources by score, plus the total row count of the table.

    The connection is closed before the extracted DataFrame is post-processed.
    """
    with sqlite_connection(db_path, read_only=True) as con:
        total_rows = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        if total_rows == 0:
            raise SystemExit(f"No rows in {db_path}.")

        if top_n > total_rows:
            print(f"Requested top {top_n:_} but the database only holds {total_rows:_} rows - using all of them.")
            top_n = total_rows

        plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT filename, score FROM results ORDER BY score DESC LIMIT ?", (top_n,)
        ).fetchall()
        print(f"Query plan: {plan[0][-1] if plan else 'unknown'}")

        df_scored = pd.read_sql(
            "SELECT score, filename FROM results ORDER BY score DESC LIMIT ?", con, params=(top_n,)
        )
    con.close()

    # Connection is now closed; the rest is pure in-memory work.
    # Scores are stored as the raw uint16 bit pattern of a float16.
    df_scored["score"] = df_scored["score"].to_numpy(dtype=np.uint16).view(np.float16).astype(np.float32)
    return df_scored, total_rows


def to_source_ids(filenames):
    """Filenames are string source IDs; strip any extension and cast to int64."""
    cleaned = filenames.astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    try:
        return cleaned.astype(np.int64)
    except ValueError as err:
        raise SystemExit(f"Could not convert filenames to integer source IDs: {err}")


def plot_counts(indices, cumul_found, n_known, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(indices, cumul_found, linestyle="--", color="black", label="N found")
    ax.axhline(n_known, color="red", label="N known anomalies")
    ax.set_xlabel("Index in ranked database")
    ax.set_ylabel("N anomalies found")
    ax.set_title("Cumulative anomalies recovered")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fractions(frac_searched, frac_found, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(frac_searched * 100, frac_found * 100, linestyle="--", color="black")
    ax.set_xlabel("Fraction of database searched (%)")
    ax.set_ylabel("Fraction of known anomalies found (%)")
    ax.set_title("Recovery vs. search depth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)

    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive.")
    if not args.known_csv.is_file():
        raise SystemExit(f"Known-anomaly CSV not found: {args.known_csv}")

    run_dir = resolve_run_dir(args.results_root, args.run)
    db_path = run_dir / "predictions.db"
    if not db_path.is_file():
        raise SystemExit(f"No predictions.db in {run_dir}")

    out_dir = args.output_dir or Path(__file__).resolve().parent / (
        f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{random.randint(1000, 9999)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory : {run_dir}")
    print(f"Database      : {db_path}")
    print(f"Output        : {out_dir}\n")

    df_known = pd.read_csv(args.known_csv, index_col=0)
    if args.id_column not in df_known.columns:
        raise SystemExit(
            f"Column '{args.id_column}' not in {args.known_csv.name}. Columns: {list(df_known.columns)}"
        )
    known_ids = set(df_known[args.id_column].astype(np.int64))
    print(f"Known anomalies: {len(known_ids):_} unique IDs from {len(df_known):_} rows")

    ensure_score_index(db_path)
    df_scored, total_rows = load_top_sources(db_path, args.top_n)
    print(f"Extracted {len(df_scored):_} of {total_rows:_} sources "
          f"(score range {df_scored['score'].min():.4f} - {df_scored['score'].max():.4f})\n")

    # Cumulative matches walking down the ranked list.
    is_match = to_source_ids(df_scored["filename"]).isin(known_ids).to_numpy()
    cumul_found = np.cumsum(is_match)
    frac_found = cumul_found / len(known_ids)

    indices = np.arange(1, len(df_scored) + 1)
    frac_searched = indices / total_rows

    plot_counts(indices, cumul_found, len(known_ids), out_dir / "anomalies_found_vs_index.png")
    plot_fractions(frac_searched, frac_found, out_dir / "fraction_found_vs_fraction_searched.png")

    pd.DataFrame(
        {
            "index": indices,
            "fraction_searched": frac_searched,
            "score": df_scored["score"].to_numpy(),
            "filename": df_scored["filename"].to_numpy(),
            "is_match": is_match,
            "cumulative_found": cumul_found,
            "fraction_found": frac_found,
        }
    ).to_csv(out_dir / "recovery_curve.csv", index=False)

    print(f"Recovered {cumul_found[-1]:_} / {len(known_ids):_} known anomalies "
          f"({frac_found[-1]:.2%}) after searching {frac_searched[-1]:.4%} of the database.")
    print(f"Wrote 2 plots and recovery_curve.csv to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
