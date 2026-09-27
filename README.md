# GAD-Net: Knowledge-Guided Adaptive Graph Network for Steel Yield-Strength Prediction

Official implementation of **GAD-Net**, a knowledge-guided adaptive graph network for yield-strength prediction in a continuous annealing production line (CAPL).

GAD-Net combines a mechanism-prior graph with data-driven topology refinement, heterogeneous graph attention, and tail-aware adaptive sample weighting. The model is designed to represent process relationships while keeping preprocessing and evaluation leakage-safe.

![GAD-Net workflow](images/fig4_workflow.jpg)

## Overview

Yield strength is a key quality indicator in continuous annealing. The process variables are physically coupled, heterogeneous, and difficult to model reliably when samples are limited or concentrated away from the boundary regions. GAD-Net uses four node groups—operating, procedure, conditional, and composition variables—and incorporates process knowledge into a globally shared graph topology.

The public release contains the main GAD-Net model and its five-fold cross-validation experiment. The industrial dataset is confidential and is therefore not included.

## Highlights

- Mechanism-prior graph construction for CAPL process variables.
- Data-driven topology refinement guided by feature relationships.
- Heterogeneous graph attention with multi-head aggregation.
- Knowledge-guided graph learning with a shared topology across samples.
- Tail-aware adaptive weighting for difficult target regions.
- Leakage-safe preprocessing fitted on each training fold.
- Five-fold cross-validation with the five manuscript seeds: `48 60 72 66 67`.

## Framework

![GAD-Net framework](images/fig7_framework.jpg)

The main training workflow is:

```text
Authorized CAPL data
        |
        v
Physical admissibility filtering
        |
        v
Stratified five-fold split
        |
        v
Training-fold-only preprocessing
        |
        v
Mechanism-prior graph + adaptive topology refinement
        |
        v
Heterogeneous graph attention and tail-aware training
        |
        v
Validation-based model selection
        |
        v
One-time evaluation on the held-out test fold
```

The application context is illustrated below.

![CAPL application scenario](images/fig5_application.jpg)

## Experimental protocol

The default experiment uses five stratified folds for each of five fixed seeds. The outer test fold contains approximately 20% of the available records; validation uses 15% of the remaining training portion. Yield-strength deciles are used for stratification. Standardization, feature statistics, and sample-weight parameters are fitted on the training portion of each fold. The validation portion is used for early stopping and model selection, and the test portion is evaluated once after training.

The runner writes one JSON file for each seed/fold and a `main_summary.json` file containing the mean, sample standard deviation, and approximate 95% confidence interval for each metric.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA driver if GPU training is required.

## Data

The raw CAPL production data are enterprise-confidential and are not distributed in this repository. To run the code, place an authorized local copy at `data/CAPL.csv`, or pass another local path with `--data`.

Do not commit the dataset, derived predictions, checkpoints, logs, or private industrial metadata to a public repository.

## Running the main experiment

From the repository root:

```bash
python main.py --data data/CAPL.csv --gpu 0 --output results_5fold_cv/main
```

Use `--gpu -1` for a CPU smoke test. The default seeds are `48 60 72 66 67`; they can be replaced by passing five integer seeds.



## Repository structure

```text
GAD-Net/
├── images/
│   ├── fig4_workflow.jpg
│   ├── fig5_application.jpg
│   └── fig7_framework.jpg
├── models/
│   ├── gad_net.py
│   └── __init__.py
├── config.py
├── data_preprocessing.py
├── fold_preprocessing.py
├── graph_construction.py
├── evaluate.py
├── train.py
├── main.py
├── requirements.txt
└── README.md
```

## Citation

If you use GAD-Net in your research, please cite the accompanying manuscript.

## License

This code is released for research use. The enterprise dataset is not included.
