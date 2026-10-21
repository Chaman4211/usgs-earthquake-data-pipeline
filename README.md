# USGS Earthquake Data Pipeline

A Python-based data engineering and machine learning project that processes real-time USGS earthquake feed data, stores transformed records in SQLite, and trains models for earthquake magnitude regression and significant-earthquake classification.

**Project:** NIAI AI/ML Program — Module 3  
**Scope:** Day 1–3 | Data Preprocessing, ETL, and Machine Learning

[GitHub Repository](https://github.com/Chaman4211/usgs-earthquake-data-pipeline) · [Jira Project](https://chamanafzaal.atlassian.net/jira/software/projects/QDP/boards/2)

## Overview

This project demonstrates an end-to-end workflow for turning earthquake feed data into structured data and machine learning outputs.

It combines data cleaning, feature engineering, preprocessing, ETL, database storage, model training, evaluation, and automated tests in one repository.

## Key Features

- **Live data ingestion:** Fetch earthquake records from the USGS all-week GeoJSON feed.
- **Data cleaning:** Handle invalid or incomplete records through the cleaning pipeline.
- **Feature engineering:** Transform raw earthquake attributes into model-ready features.
- **Preprocessing:** Prepare data for training and testing while avoiding target leakage.
- **ETL pipeline:** Extract, transform, and load earthquake records with logging.
- **SQLite storage:** Persist transformed earthquake records in a local database.
- **Machine learning:** Train regression and classification models with scikit-learn.
- **Model evaluation:** Compare models with a dummy baseline and evaluate predictive performance.
- **Testing:** Run automated tests for pipeline, model, and evaluation functionality.

## Pipeline Architecture

```text
USGS Earthquake Feed
        |
        v
   Data Extraction
        |
        v
 Cleaning and Validation
        |
        v
 Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
 Preprocessing             ETL Pipeline
 Train/Test Split               |
        |                       v
        |                  SQLite Database
        v
 Model Training
        |
        v
 Model Evaluation
        |
        v
 Saved Models and Reports
```

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Data storage | SQLite |
| Serialization | Joblib |
| Testing | pytest |
| Development | Jupyter Notebook, Git, GitHub |
| Environment | Conda |

## Project Structure

```text
day1-quake-preprocessing/
├── quakes/              # ETL, cleaning, features, models and training
├── tests/               # Automated tests
├── notebooks/           # Scikit-learn learning notebook
├── data/
│   ├── processed/       # Processed CSV files, database and preprocessor
│   └── stream/          # Stream input data
├── models/              # Saved machine learning models
├── reports/             # Evaluation reports
├── logs/                # ETL logs
├── environment.yml      # Conda environment specification
├── ANSWERS.md            # Day 1 task answers
├── ANSWERS_ML.md         # Day 2–3 machine learning answers
└── JIRA_WORK_ITEMS.md    # Jira task documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chaman4211/usgs-earthquake-data-pipeline.git
cd usgs-earthquake-data-pipeline
```

### 2. Create the Conda environment

If you have Conda installed:

```bash
conda env create -f environment.yml
conda activate niai-ml
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
conda activate niai-ml
```

### 3. Run the ETL pipeline

Fetch the current USGS feed, transform the records, and load the results into SQLite:

```bash
python -m quakes.etl --mode batch
```

To run the merge mode, which also incorporates stream input data:

```bash
python -m quakes.etl --mode merge
```

Successful execution logs the extraction, transformation, loading, and completion stages.

## Machine Learning

### Regression — Earthquake Magnitude

Train a linear regression model:

```bash
python -m quakes.train --task regression --model linear
```

Train a Ridge regression model:

```bash
python -m quakes.train --task regression --model ridge --alpha 1
```

Train a polynomial regression model:

```bash
python -m quakes.train --task regression --model poly --degree 3 --alpha 10
```

### Classification — Significant Earthquakes

Train a logistic regression classifier:

```bash
python -m quakes.train --task classification --model logistic
```

Train with class balancing enabled:

```bash
python -m quakes.train --task classification --model logistic --balanced
```

The training workflow evaluates the selected model against a dummy baseline and uses training-data cross-validation for model assessment and threshold selection where implemented.

**Note:** Running a training command again can overwrite the corresponding saved model and report.

## Generated Outputs

| Output | Purpose |
|---|---|
| `data/processed/train.csv` | Training dataset |
| `data/processed/test.csv` | Held-out test dataset |
| `data/processed/preprocessor.joblib` | Saved preprocessing object |
| `data/processed/quakes.db` | SQLite database |
| `data/stream/stream.jsonl` | Stream input records |
| `models/` | Serialized machine learning models |
| `reports/` | Model evaluation reports |
| `logs/etl.log` | ETL execution logs |

Generated files depend on which pipeline and training commands have been run.

## Testing

Run the complete test suite:

```bash
pytest -q
```

Run individual test groups:

```bash
pytest -q tests/test_etl.py
pytest -q tests/test_models.py
pytest -q tests/test_evaluate.py
```

Tests help verify the ETL, model creation, and evaluation functionality.

## Data Quality and Model Reliability

- Separate training and test data to support evaluation on held-out examples.
- Exclude target and other leakage-prone fields from model features.
- Fit preprocessing steps on training data rather than using information from the test set.
- Compare model performance against a dummy baseline.
- Use cross-validation on training data where implemented.

## Project Management

Development work is tracked in Jira.

- [Day 1 — Data Preprocessing Epic (QDP-1)](https://chamanafzaal.atlassian.net/browse/QDP-1)
- [Day 2–3 — ETL and Machine Learning Epic (QDP-13)](https://chamanafzaal.atlassian.net/browse/QDP-13)
- [GitHub Repository](https://github.com/Chaman4211/usgs-earthquake-data-pipeline)

## Learning Outcomes

This project demonstrates practical experience with:

- Building a structured data processing pipeline.
- Cleaning and transforming real-world data.
- Working with ETL, logging, and relational data storage.
- Preparing data for machine learning while considering leakage.
- Training and evaluating regression and classification models.
- Using GitHub, Jira, Conda, and automated tests in a project workflow.

## Data Source

Earthquake data is provided by the [United States Geological Survey (USGS)](https://earthquake.usgs.gov/earthquakes/feed/).

---

**Project status:** Day 1–3 implementation and testing.

