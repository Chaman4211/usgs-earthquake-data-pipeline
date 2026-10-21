# USGS Earthquake Data Pipeline — Day 2–3 Lab

A beginner-friendly earthquake data pipeline that fetches USGS earthquake records, cleans and engineers features, stores the results in SQLite, and trains machine-learning models.

## Requirements

- Conda
- Python 3.11
- Internet connection for the live USGS feed

## Environment setup

```bash
conda env update -f environment.yml --prune
conda activate niai-ml
```

## Run the ETL pipeline

Batch mode fetches the USGS feed, applies cleaning and feature engineering, and upserts records into SQLite.

```bash
python -m quakes.etl --mode batch
```

Merge mode includes available stream JSONL files:

```bash
python -m quakes.etl --mode merge
```

Database: `data/processed/quakes.db`

Logs: `logs/etl.log`

## Train regression models

```bash
python -m quakes.train --task regression --model linear
python -m quakes.train --task regression --model ridge --alpha 1
python -m quakes.train --task regression --model poly --degree 3 --alpha 10
```

## Train classification models

```bash
python -m quakes.train --task classification --model logistic
python -m quakes.train --task classification --model logistic --balanced
```

The training script compares the selected model with a dummy baseline. It evaluates the test split and performs cross-validation on training data. Classification threshold selection uses out-of-fold training predictions.

## Output files

- `models/`: serialized trained models (`.joblib`)
- `reports/`: JSON evaluation reports
- `data/processed/`: processed data and SQLite database
- `logs/etl.log`: ETL log

Note: running the same model and task again overwrites its existing model and report files.

## Tests

```bash
pytest -q
pytest -q tests/test_etl.py
pytest -q tests/test_models.py
pytest -q tests/test_evaluate.py
```

## Schedule hourly ETL on Linux

1. Find the absolute path to your project:

   ```bash
   pwd
   ```

2. Find the Conda executable:

   ```bash
   which conda
   ```

3. Open your crontab:

   ```bash
   crontab -e
   ```

4. Add an hourly entry, replacing the example paths with your actual paths:

   ```cron
   0 * * * * cd /home/YOUR_USER/path/to/day1-quake-preprocessing && /home/YOUR_USER/anaconda3/envs/niai-ml/bin/python -m quakes.etl --mode batch >> logs/cron.log 2>&1
   ```

This runs ETL at the start of each hour. Ensure the project path and Python executable are correct. The computer must be on, and the network must be available.

## Schedule hourly ETL on Windows

1. Open **Task Scheduler** and choose **Create Basic Task**.
2. Name it `USGS Earthquake ETL`.
3. Select **Daily**, then configure the task to repeat every **1 hour** for **1 day** in the advanced trigger settings.
4. Set the action to **Start a program**.
5. Set **Program/script** to the full path of the Python executable inside your `niai-ml` environment.
6. Set **Add arguments** to:

   ```text
   -m quakes.etl --mode batch
   ```

7. Set **Start in** to the full project directory.
8. Save the task and use Task Scheduler's **Run** action to test it.

These scheduling examples are instructions only; no scheduled task is created automatically.

## Notebook

Open the Day 2–3 notebook with:

```bash
jupyter lab
```

Notebook: `notebooks/day2-3_part1_sklearn_basics.ipynb`

Complete all cells and answer the notebook questions before submitting.
