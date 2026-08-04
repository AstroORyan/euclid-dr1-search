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
  `predictions.db`; pass `--resume` to continue an interrupted run. A stale
  scratch copy under `<tmpdir>/anomaly_match_db/runN/` never blocks a run — it is
  deleted at the start of a fresh one, since a relocated run would otherwise
  seed itself from it and mix another seed's scores into the results.
- `predictions.db` is pinned to the run directory. Left to itself, AnomalyMatch
  relocates the *live* database to `<tmpdir>/anomaly_match_db/runN/` whenever the
  session directory is on NFS and snapshots it back after every chunk, keeping
  two full copies of a 50M-row database — awkward on Datalabs. The relocation
  guards against write-ahead-log bloat under a slow reader (the UI polling over
  NFS), which does not apply headless; `--scratch-db` restores it if a run ever
  does show WAL growth.
- AnomalyMatch writes scratch files (`tmp/`, `anomaly_match_results/`) relative
  to the working directory, so the script `chdir`s into `--work-dir` (the
  results root by default) to keep them out of the repo.
- `--build-score-index` builds the `idx_score_covering` index at the end of each
  run, which the analysis then does not have to build itself.

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
