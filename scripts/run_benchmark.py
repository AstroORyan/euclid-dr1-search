"""Run repeated AnomalyMatch 2.0 benchmarking runs over the Euclid Cutana catalogues.

Each run trains a fresh model on the *same* labelled CSV and then scores every
source in the Cutana catalogues (~50M). The only thing that changes between runs
is `cfg.seed`, which reseeds the unlabelled sampler: with a fixed label set the
Cutana source derives its RNG from `cfg.seed` (mixed with a hash of the labelled
IDs), so a different seed draws a different set of tiles / unlabelled cutouts
while every other knob stays identical.

Per run the script:
  1. Builds an identical config (channel combination, asinh normalisation,
     3-band output) apart from the seed, and opens an AnomalyMatch Session.
  2. Validates the labels against the catalogues and builds/reuses the labelled
     cutout cache next to the label CSV.
  3. Trains via `subprocess_scripts/training_process.py` (launched by
     `BackendInterface.launch_training_subprocess`).
  4. Scores every source via `BackendInterface.evaluate_all_images`, which
     streams the catalogues in chunks through the Cutana prediction subprocess.
  5. Snapshots and closes `predictions.db` (WAL checkpointed, sidecars gone)
     before moving on to the next run directory.

Outputs land in `<results-root>/run1 ... runN`, which is the layout
`benchmark_analysis.py` expects.

Example
-------
    conda run -n am python run_benchmark.py --runs 4
    conda run -n am python run_benchmark.py --seeds 42,1337 --num-train-iter 300
"""

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger

import anomaly_match as am
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.prediction import (
    AnomalyScoreDB,
    is_relocated,
    prediction_db_path,
    session_db_path,
    snapshot_db_to_session,
)
from anomaly_match_ui.utils.backend_interface import BackendInterface

WORKSPACE = Path("/media/team_workspaces/AnomalyMatch-IDR1-Search/benchmarking_tests")
DEFAULT_DATA_DIR = WORKSPACE / "data" / "source_cats"
DEFAULT_LABEL_FILE = WORKSPACE / "data" / "labelled_lenses" / "labelled_data.csv"
DEFAULT_RESULTS_ROOT = WORKSPACE / "results"

# Seeds are fixed (not drawn at random) so every run is reproducible and the
# seed that produced a given run directory is recoverable from run_config.json.
DEFAULT_SEEDS = (42, 1337, 2718, 31415)

# Cutana catalogues carry 4 bands, logged in this order:
#   0: NIR-H   1: NIR-J   2: NIR-Y   3: VIS
# We only use 3 of them: R = NIR-Y, G = 0.5*NIR-Y + 0.5*VIS, B = VIS.
# Rows are output channels, columns are input bands.
CHANNEL_COMBINATION = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
N_OUTPUT_CHANNELS = 3

# Asinh with fitsbolt's default scaling and a 99.5th-percentile clip per channel.
ASINH_SCALE = [0.7, 0.7, 0.7]
ASINH_CLIP = [99.5, 99.5, 99.5]

DEFAULT_IMAGE_SIZE = 192
DEFAULT_NUM_TRAIN_ITER = 300
DEFAULT_TOP_N = 5_000
DEFAULT_CHUNK_SIZE = 500_000

SCORE_INDEX_NAME = "idx_score_covering"

# How often to echo training progress while the subprocess grinds through
# iterations, in seconds.
TRAIN_LOG_INTERVAL = 30.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory of Cutana source catalogues (*.parquet). Used for both training and scoring.",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=DEFAULT_LABEL_FILE,
        help="Labelled data CSV (id,label). Identical for every run.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory; each run writes to <results-root>/runN.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds, one per run (default: %(default)s).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of runs to execute (default: all seeds).",
    )
    parser.add_argument(
        "--first-run-index",
        type=int,
        default=1,
        help="Run number of the first run, so a second invocation can write run5 onwards.",
    )
    parser.add_argument("--num-train-iter", type=int, default=DEFAULT_NUM_TRAIN_ITER)
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square cutout resolution fed to the network.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top-scoring sources tracked during scoring.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Sources per prediction subprocess chunk (cfg.subprocess_buffer_size).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Directory the script runs from. AnomalyMatch writes scratch files "
            "(tmp/, anomaly_match_results/) relative to the CWD. "
            "Defaults to the results root."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Allow a run directory that already holds a predictions.db. Scoring "
            "then skips sources already present in the database."
        ),
    )
    parser.add_argument(
        "--build-score-index",
        action="store_true",
        help=f"Build the {SCORE_INDEX_NAME} index after each run (slow, but speeds up analysis).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Score only, reusing the model already in <run-dir>/iteration_0/model.safetensors.",
    )
    return parser.parse_args(argv)


def build_config(args, seed, run_dir):
    """Build the AnomalyMatch config for one run.

    Everything here is identical across runs except `seed`. Note that the paths
    set here are only the *inputs* - `Session.__init__` rewrites `output_dir`,
    `save_dir` and `model_path` to its own timestamped session folder, so
    `pin_output_dir` puts them back onto the run directory afterwards.
    """
    cfg = am.get_default_cfg()

    cfg.name = f"benchmark_run_seed{seed}"
    cfg.log_level = "INFO"
    cfg.seed = seed

    # Data: the same catalogues are both the training pool and the search space.
    cfg.data_dir = str(args.data_dir)
    cfg.prediction_search_dir = str(args.data_dir)
    cfg.label_file = str(args.label_file)
    cfg.training_data_source = DataSourceType.CUTANA
    cfg.test_ratio = 0.0

    # Normalisation - the part that must not drift between runs.
    cfg.normalisation.normalisation_method = am.NormalisationMethod.ASINH
    cfg.normalisation.norm_asinh_scale = list(ASINH_SCALE)
    cfg.normalisation.norm_asinh_clip = list(ASINH_CLIP)
    cfg.normalisation.channel_combination = CHANNEL_COMBINATION.copy()
    cfg.normalisation.n_output_channels = N_OUTPUT_CHANNELS
    cfg.num_channels = N_OUTPUT_CHANNELS
    cfg.normalisation.image_size = [args.image_size, args.image_size]
    cfg.normalisation.fits_extension = None
    cfg.normalisation.apply_flux_conversion = True
    cfg.normalisation.cutout_padding_factor = 1.0

    # Training / streaming.
    cfg.num_train_iter = args.num_train_iter
    cfg.top_N = args.top_n
    cfg.subprocess_buffer_size = args.chunk_size
    cfg.cutana_stratify_source_size = True

    # Leave model_path unset: the session assigns it, then the training launcher
    # points it at <run_dir>/iteration_N/model.safetensors.
    cfg.model_path = None
    cfg.output_dir = str(run_dir)
    cfg.save_dir = str(run_dir)
    return cfg


def pin_output_dir(cfg, run_dir):
    """Point the session's outputs back at `run_dir`.

    `Session.__init__` redirects `output_dir`/`save_dir` to
    `anomaly_match_results/sessions/<name>_<timestamp>/`. We want one clean
    directory per run instead, and `benchmark_analysis.py` expects
    `<results-root>/runN/predictions.db`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(run_dir)
    cfg.save_dir = str(run_dir)
    cfg.model_path = None


def check_run_dir(cfg, run_dir, resume):
    """Refuse to overwrite an existing run unless `--resume` was passed.

    Checks both the run directory and, on a networked session dir, the local
    scratch copy the live database is relocated to - a leftover scratch DB from
    an earlier invocation would silently be resumed (and mixed with a different
    seed's scores).
    """
    for label, path in (
        ("session", Path(session_db_path(cfg))),
        ("live", Path(prediction_db_path(cfg))),
    ):
        if path.is_file() and not resume:
            raise SystemExit(
                f"A {label} predictions.db already exists at {path}.\n"
                f"Delete it, choose another run index, or pass --resume to continue it."
            )
    if resume:
        logger.warning("Resume enabled - existing scores in {} will be kept", run_dir)


def validate_labels(args):
    """Match the labelled IDs against the catalogues once, up front.

    The result is reused for every run: the label CSV never changes, so the
    validation (a scan of every catalogue) only needs doing once.
    """
    logger.info("Validating labels in {} against {}", args.label_file, args.data_dir)
    message, found_locations, partial = BackendInterface.validate_labels_against_source(
        str(args.data_dir), DataSourceType.CUTANA, str(args.label_file)
    )
    if partial:
        raise SystemExit("Label validation was interrupted before completing.")
    logger.info("Label validation: {}", message)
    if not found_locations:
        raise SystemExit(f"None of the labelled IDs in {args.label_file} were found in {args.data_dir}.")
    return found_locations


def prepare_labeled_cache(args, found_locations):
    """Build (or reuse) the cache of labelled cutouts.

    The cache lives next to the label CSV and is keyed on the label set plus the
    extraction parameters, so it is built during the first run and reused by the
    rest. It is also what puts the Cutana source into lazy mode - without it,
    training would try to index all ~50M sources.
    """
    cache_dir = BackendInterface.build_labeled_cache(
        found_locations, str(args.label_file), DataSourceType.CUTANA
    )
    if not cache_dir:
        raise SystemExit("Failed to build the labelled data cache.")
    logger.info("Labelled cutout cache: {}", cache_dir)
    return cache_dir


def drain_stderr(stream, prefix):
    """Consume a subprocess pipe into the log so it can never fill and block."""
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            logger.info("[{}] {}", prefix, line)
    stream.close()


def read_progress(path, offset):
    """Read new JSON-lines records from the training progress file.

    Returns the parsed records plus the new byte offset. Partial trailing lines
    are left for the next call.
    """
    records = []
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
        offset += len(data)
    text = data.decode("utf-8", errors="replace")
    if text and not text.endswith("\n"):
        # Keep the incomplete tail for the next read.
        cut = text.rfind("\n") + 1
        offset -= len(text[cut:].encode("utf-8"))
        text = text[:cut]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed progress line: {}", line[:200])
    return records, offset


def train(num_train_iter):
    """Launch the training subprocess and block until it finishes.

    Returns the path of the saved checkpoint.
    """
    process, temp_dir, progress_file = BackendInterface.launch_training_subprocess(
        num_train_iter=num_train_iter
    )
    logger.info("Training subprocess started (pid={})", process.pid)

    stderr_thread = threading.Thread(target=drain_stderr, args=(process.stderr, "train"), daemon=True)
    stderr_thread.start()

    offset = 0
    model_path = None
    last_log = 0.0
    start = time.time()
    try:
        while True:
            if os.path.isfile(progress_file):
                records, offset = read_progress(progress_file, offset)
                for record in records:
                    status = record.get("status")
                    if status == "training":
                        now = time.time()
                        if now - last_log >= TRAIN_LOG_INTERVAL:
                            logger.info(
                                "Training iteration {}/{}",
                                record.get("iteration"),
                                record.get("total"),
                            )
                            last_log = now
                    elif status == "done":
                        model_path = record.get("model_path")
                        logger.info("Training reported done after {:.1f}s", record.get("elapsed", 0))
                    elif status in ("loading", "saving"):
                        logger.info("Training: {}", record.get("message", status))
                    elif status == "error":
                        logger.error("Training error: {}", record)
            if process.poll() is not None:
                break
            time.sleep(1.0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        stderr_thread.join(timeout=10)

    # Flush whatever the subprocess wrote between the last poll and exit.
    if os.path.isfile(progress_file):
        records, offset = read_progress(progress_file, offset)
        for record in records:
            if record.get("status") == "done":
                model_path = record.get("model_path")

    if process.returncode != 0:
        raise RuntimeError(
            f"Training subprocess exited with code {process.returncode}. "
            f"See the training.log next to the checkpoint and {temp_dir}."
        )
    if not model_path:
        raise RuntimeError(f"Training never reported a checkpoint path. Progress file: {progress_file}")

    logger.info("Training finished in {} - checkpoint {}", elapsed(start), model_path)
    return model_path


def score_all(cfg, top_n):
    """Score every source in the search directory.

    Returns the skip statistics reported by the session so a partially failed
    run (e.g. the data volume dropping out mid-run) is visible in the summary.
    """
    start = time.time()
    logger.info("Scoring every source in {}", cfg.prediction_search_dir)
    BackendInterface.evaluate_all_images(top_n)
    stats = BackendInterface.get_last_run_skip_stats()
    logger.info("Scoring finished in {}", elapsed(start))
    if stats.get("skipped_chunks"):
        logger.warning(
            "{} chunk(s) skipped, ~{:,} source(s) unscored",
            stats["skipped_chunks"],
            stats.get("skipped_sources", 0),
        )
    return stats


def row_count(db_path):
    """Number of scored rows, via MAX(id) rather than COUNT(*).

    `results.id` is an autoincrementing primary key on an append-only table, so
    MAX(id) is the row count but costs an index lookup instead of a full scan of
    a 50M-row table.
    """
    if not os.path.isfile(db_path):
        return 0
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT COALESCE(MAX(id), 0) FROM results").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error as err:
        logger.warning("Could not count rows in {}: {}", db_path, err)
        return 0
    finally:
        con.close()


def build_score_index(db_path):
    """Create the covering (score DESC, filename) index used by the analysis."""
    logger.info("Building {} on {} (this can take a while)", SCORE_INDEX_NAME, db_path)
    start = time.time()
    con = sqlite3.connect(db_path)
    try:
        con.execute(f"CREATE INDEX IF NOT EXISTS {SCORE_INDEX_NAME} ON results (score DESC, filename)")
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    logger.info("Index built in {}", elapsed(start))


def close_database(cfg, build_index):
    """Checkpoint, close and snapshot the predictions database.

    Opening `AnomalyScoreDB` as a context manager runs
    `PRAGMA wal_checkpoint(TRUNCATE)` and `PRAGMA optimize` on the way out, which
    folds the write-ahead log back into the `.db` file and removes the
    `-wal`/`-shm` sidecars. Only then is the database copied back into the
    session directory, so the durable copy is a single self-contained file.
    """
    live_path = prediction_db_path(cfg)
    if not os.path.isfile(live_path):
        logger.error("No predictions.db at {} - the run produced no scores.", live_path)
        return {"rows": 0, "db_path": live_path}

    n_rows = row_count(live_path)
    logger.info("Closing predictions.db at {} ({:,} rows)", live_path, n_rows)

    if build_index:
        build_score_index(live_path)

    with AnomalyScoreDB(live_path) as db:
        # The context manager's exit closes the connection and checkpoints.
        logger.debug("Opened {} for the closing checkpoint", db.path)

    if is_relocated(cfg):
        logger.info("Snapshotting the live database back to {}", cfg.output_dir)
        snapshot_db_to_session(cfg)

    final_path = session_db_path(cfg)
    leftovers = [suffix for suffix in ("-wal", "-shm") if os.path.isfile(f"{final_path}{suffix}")]
    if leftovers:
        logger.warning(
            "Database sidecars still present after close: {}",
            ", ".join(f"{final_path}{s}" for s in leftovers),
        )
    else:
        logger.info("Database closed cleanly, no -wal/-shm sidecars left")

    return {
        "rows": n_rows,
        "db_path": final_path,
        "db_bytes": os.path.getsize(final_path) if os.path.isfile(final_path) else 0,
    }


def elapsed(start):
    """Human-readable elapsed time since `start` (a `time.time()` stamp)."""
    return str(timedelta(seconds=int(time.time() - start)))


def run_once(args, run_index, seed, found_locations):
    """Set up, train, score and tear down a single seeded run."""
    run_dir = args.results_root / f"run{run_index}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_id = logger.add(
        run_dir / "benchmark_run.log",
        rotation="50 MB",
        format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
        level="DEBUG",
    )
    start = time.time()
    logger.info("=" * 70)
    logger.info("Run {} - seed {} -> {}", run_index, seed, run_dir)
    logger.info("=" * 70)

    session = None
    summary = {
        "run": f"run{run_index}",
        "seed": seed,
        "run_dir": str(run_dir),
        "started": datetime.now().isoformat(timespec="seconds"),
        "status": "failed",
    }

    try:
        cfg = build_config(args, seed, run_dir)
        session = am.Session(cfg)
        # Session.__init__ hijacks the output paths; put them back on run_dir.
        pin_output_dir(cfg, run_dir)
        BackendInterface.set_session(session)

        check_run_dir(cfg, run_dir, args.resume)
        prepare_labeled_cache(args, found_locations)

        # Record exactly what this run was configured with, next to its results.
        write_run_config(cfg, run_dir, args, seed)

        if args.skip_training:
            model_path = run_dir / "iteration_0" / "model.safetensors"
            if not model_path.is_file():
                raise SystemExit(f"--skip-training was passed but {model_path} does not exist.")
            cfg.model_path = str(model_path)
            logger.info("Skipping training, scoring with {}", model_path)
        else:
            model_path = train(args.num_train_iter)
        summary["model_path"] = str(model_path)

        skip_stats = score_all(cfg, args.top_n)
        summary.update(
            {
                "skipped_chunks": skip_stats.get("skipped_chunks", 0),
                "skipped_sources": skip_stats.get("skipped_sources", 0),
                "total_chunks": skip_stats.get("total_chunks", 0),
                "total_sources": skip_stats.get("total_sources", 0),
            }
        )
        summary["status"] = "completed"
    except KeyboardInterrupt:
        logger.warning("Interrupted - stopping the prediction subprocess before closing the DB")
        if session is not None:
            session.request_prediction_stop()
        summary["status"] = "interrupted"
        raise
    finally:
        # Always leave the database in a closed, self-contained state, even if
        # the run failed part-way: the scores written so far are still useful.
        if session is not None:
            try:
                summary.update(close_database(session.cfg, args.build_score_index))
            except Exception:
                logger.exception("Failed to close the predictions database cleanly")
            BackendInterface.set_session(None)
        summary["duration"] = elapsed(start)
        summary["finished"] = datetime.now().isoformat(timespec="seconds")
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Run {} {} in {}", run_index, summary["status"], summary["duration"])
        logger.remove(log_id)

    return summary


def write_run_config(cfg, run_dir, args, seed):
    """Save the parameters that define this run, for the record."""
    record = {
        "seed": seed,
        "data_dir": cfg.data_dir,
        "label_file": cfg.label_file,
        "labeled_cache_path": cfg.labeled_cache_path,
        "output_dir": cfg.output_dir,
        "num_train_iter": cfg.num_train_iter,
        "image_size": list(cfg.normalisation.image_size),
        "normalisation_method": str(cfg.normalisation.normalisation_method),
        "norm_asinh_scale": list(cfg.normalisation.norm_asinh_scale),
        "norm_asinh_clip": list(cfg.normalisation.norm_asinh_clip),
        "channel_combination": CHANNEL_COMBINATION.tolist(),
        "n_output_channels": cfg.normalisation.n_output_channels,
        "apply_flux_conversion": cfg.normalisation.apply_flux_conversion,
        "cutana_stratify_source_size": cfg.cutana_stratify_source_size,
        "subprocess_buffer_size": cfg.subprocess_buffer_size,
        "top_N": cfg.top_N,
        "anomaly_match_version": am.__version__,
        "command": " ".join(sys.argv),
    }
    (run_dir / "run_config.json").write_text(json.dumps(record, indent=2))


def main(argv=None):
    args = parse_args(argv)

    args.data_dir = args.data_dir.expanduser().resolve()
    args.label_file = args.label_file.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Catalogue directory not found: {args.data_dir}")
    if not args.label_file.is_file():
        raise SystemExit(f"Label CSV not found: {args.label_file}")

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    if args.runs is not None:
        if args.runs > len(seeds):
            raise SystemExit(f"--runs {args.runs} exceeds the {len(seeds)} seeds provided.")
        seeds = seeds[: args.runs]
    if not seeds:
        raise SystemExit("No seeds given.")

    args.results_root.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or args.results_root).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    # AnomalyMatch writes tmp/ buffers and its own session folders relative to
    # the working directory, so keep that inside the benchmark area.
    os.chdir(work_dir)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        args.results_root / "benchmark.log",
        rotation="50 MB",
        format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
        level="INFO",
    )

    logger.info("AnomalyMatch {} - {} run(s), seeds {}", am.__version__, len(seeds), seeds)
    logger.info("Catalogues : {}", args.data_dir)
    logger.info("Labels     : {}", args.label_file)
    logger.info("Results    : {}", args.results_root)
    logger.info("Working dir: {}", work_dir)

    found_locations = validate_labels(args)

    summaries = []
    overall_start = time.time()
    for offset, seed in enumerate(seeds):
        run_index = args.first_run_index + offset
        summaries.append(run_once(args, run_index, seed, found_locations))
        # Write the master summary after every run so a crash later does not
        # cost the results of the runs that already finished.
        (args.results_root / "benchmark_summary.json").write_text(json.dumps(summaries, indent=2))

    logger.info("All {} run(s) finished in {}", len(summaries), elapsed(overall_start))
    for summary in summaries:
        logger.info(
            "{} (seed {}): {} - {:,} rows in {}",
            summary["run"],
            summary["seed"],
            summary["status"],
            summary.get("rows", 0),
            summary["duration"],
        )


if __name__ == "__main__":
    sys.exit(main())
