# euclid-dr1-search

Repository for searching Euclid for astrophysical anomalies.

The current focus is benchmarking **AnomalyMatch 2.0** against the Euclid IDR1
Cutana source catalogues (~50M sources), using a fixed set of labelled
gravitational lenses and a catalogue of known lenses to measure recovery.

## Layout

```
scripts/     benchmarking and analysis scripts
notebooks/   utility notebooks for data setup and exploration
data/        labelled lenses, known lenses, Cutana source catalogues
results/     one directory per benchmarking run (predictions.db, logs, config)
```

## Scripts

### `scripts/run_benchmark.py`

Runs several seeded AnomalyMatch 2.0 benchmarking runs back to back. Each run
trains a fresh model on the *same* labelled CSV and then scores every source in
the Cutana catalogues. The only thing that changes between runs is `cfg.seed`,
which reseeds the unlabelled sampler: with a fixed label set, the Cutana source
derives its RNG from `cfg.seed` mixed with a hash of the labelled IDs, so a
different seed draws a different set of tiles and unlabelled cutouts while every
other parameter stays identical.

Per run the script:

1. Builds an identical config apart from the seed and opens an AnomalyMatch
   `Session`, driving it through `BackendInterface` (no UI).
2. Validates the labels against the catalogues (done once, up front, since the
   label CSV never changes) and builds or reuses the labelled cutout cache next
   to the label CSV. That cache is also what keeps the Cutana source in lazy
   mode — without it, training would try to index all ~50M sources.
3. Trains via `subprocess_scripts/training_process.py`.
4. Scores every source via `evaluate_all_images`, which streams the catalogues
   in chunks through the Cutana prediction subprocess.
5. Checkpoints and closes `predictions.db` (WAL folded back in, no `-wal`/`-shm`
   sidecars left behind), snapshotting it back from local scratch if the live
   database was relocated there, before moving on to the next run.

Fixed configuration shared by every run:

| Setting | Value |
|---|---|
| `channel_combination` | `[[0,0,1,0], [0,0,0.5,0.5], [0,0,0,1]]` |
| Bands | 3 of the 4 Cutana bands (`NIR-H, NIR-J, NIR-Y, VIS`) → R = NIR-Y, G = ½NIR-Y + ½VIS, B = VIS |
| Normalisation | `ASINH`, scale `[0.7, 0.7, 0.7]` (default), clip `[99.5, 99.5, 99.5]` |
| `n_output_channels` | 3 |
| `apply_flux_conversion` | `True` (Euclid `MAGZERO`) |
| Image size | 192 × 192 |
| Training iterations | 300 |
| `top_N` | 5000 |
| Seeds | 42, 1337, 2718, 31415 (one per run) |

Everything in that table is overridable from the command line. Results land in
`<results-root>/run1 … runN`, each containing `predictions.db`,
`run_config.json` (the exact parameters used, including the seed),
`run_summary.json`, `benchmark_run.log` and the trained checkpoint under
`iteration_0/`.

```bash
# All four runs with the default seeds
conda run -n am python scripts/run_benchmark.py --runs 4

# Smoke test before committing to the full sweep
conda run -n am python scripts/run_benchmark.py --seeds 42 --runs 1 --num-train-iter 5
```

Notes:

- The script refuses to start if the *run directory* already holds a
  `predictions.db`; pass `--resume` to continue an interrupted run. All
  validation runs before anything is written, so a rejected run leaves no run
  directory, no log file and no session folder behind.
- `--build-score-index` builds the `idx_score_covering` index at the end of each
  run, which the analysis then does not have to build itself.

#### Disk hygiene

Left to its own devices, AnomalyMatch spreads a run across several places and
cleans up none of them, which adds up quickly at 50M sources on a Datalabs pod.
The script is deliberately aggressive about pulling everything back into
`<results-root>/runN/`:

- **No second copy of the database.** `prediction_db_dir` is pinned to the run
  directory. Otherwise AnomalyMatch relocates the *live* database to
  `<tmpdir>/anomaly_match_db/runN/` whenever the session directory is on NFS and
  snapshots it back after every chunk, leaving two full copies of a 50M-row
  database. That relocation guards against write-ahead-log bloat under a slow
  reader (the UI polling over NFS), which does not apply headless — `--scratch-db`
  restores it if a run ever does show WAL growth.
- **Stale scratch databases are deleted, not obeyed.** A leftover
  `<tmpdir>/anomaly_match_db/runN/` never blocks a run; it is removed (with its
  size logged) at the start of a fresh one, since a relocated run would otherwise
  seed itself from it and mix another seed's scores into the results.
- **No timestamped session folders.** Every `Session` creates
  `anomaly_match_results/sessions/<name>_<timestamp>/` and logs into it, even
  though the run's outputs are re-pointed at the run directory. The script
  removes that folder — and, importantly, its log sinks — once the run finishes.
  Without that, the sinks would stay registered for the life of the process and
  every later run would keep writing into every earlier run's folder.
- **Scratch files stay out of the repo.** AnomalyMatch writes `tmp/` and
  `anomaly_match_results/` relative to the working directory, so the script
  `chdir`s into `--work-dir` (the results root by default).

Clearing out session folders left by *earlier* work is a manual job, and worth a
look before deleting: the ones this script creates only ever contain
`session.log`, but a session folder from a UI run is the real output directory
and holds its model, labels and predictions.

### `scripts/benchmark_analysis.py`

Takes a completed run's `predictions.db`, ranks every source by anomaly score
and measures how many of the known lenses are recovered as a function of search
depth. Writes a recovery curve (CSV) plus the cumulative-count and
fraction-found figures.

```bash
python scripts/benchmark_analysis.py --known-csv data/known_lenses/known_lenses_in_sample.csv --run run1
```

## Notebooks

- **`benchmark_ui.ipynb`** — testing the benchmarking approach through the
  AnomalyMatch UI, i.e. the interactive equivalent of what `run_benchmark.py`
  now does headlessly.
- **`benchmark-nb.ipynb`** — prototype for the benchmarking figures that the
  scripts produce: reads `predictions.db`, matches the top-ranked sources
  against the known lenses and plots the recovery curves.
- **`notebooks/`** — utility notebooks for preparing the data used by the
  benchmarking and analysis:
  - `finding-known-lenses.ipynb` — assembles the catalogue of known lenses
    present in the sample.
  - `identifying-missing-labels.ipynb` — tracks down labelled sources that do
    not appear in the Cutana catalogues.

## Development checks

Two GitHub Actions workflows run on every push and pull request, mirroring the
equivalents in the AnomalyMatch repository:

| Workflow | Checks |
|---|---|
| `.github/workflows/formatting.yml` | `ruff check` (pycodestyle, pyflakes, import sorting) and `ruff format --check` |
| `.github/workflows/dead_code.yml` | `vulture` at 100% confidence (blocking) and at 60% (required) |

Both are configured in `pyproject.toml`: line length 110, Python 3.11, notebooks
excluded — they are exploratory working documents, and holding them to the
script standard would mean reformatting cells on every commit. Licence-header
checks and pytest runs are deliberately *not* included.

To run them locally (`pip install ruff "vulture>=2.10"` if needed):

```bash
ruff check .                 # add --fix to apply the safe fixes
ruff format .                # --check to report without rewriting
python -m vulture scripts/ .vulture_whitelist.py --min-confidence 60
```

Ruff is pinned to the same version in CI as the one used locally, so a new ruff
release cannot fail a repository that was clean when it was pushed; bump both
together.

Vulture reports anything it cannot see being used. Most false positives here are
AnomalyMatch config attributes: the scripts only ever *write* them onto the
config, and they are read inside AnomalyMatch or its subprocesses. Add those to
`.vulture_whitelist.py` with a comment saying who consumes them — a genuinely
unused attribute is worth deleting instead.
