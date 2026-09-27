# GAD-Net: Knowledge-Guided Adaptive Graph Network for Steel Yield-Strength Prediction

Official implementation of **GAD-Net**, a knowledge-guided heterogeneous graph attention network for predicting the yield strength of cold-rolled strip steel in continuous annealing production lines (CAPLs).

![GAD-Net workflow](images/fig4_workflow.jpg)

## Overview

Yield strength is a key mechanical property of cold-rolled strip steel, but it is normally obtained by offline tensile testing. GAD-Net estimates yield strength from available CAPL production variables before the destructive test result is returned. The model represents operating, procedure, conditional, and composition variables as heterogeneous graph nodes, then combines mechanistic priors with adaptive graph learning and relation-specific attention.

The public code release contains the model implementation and the repeated stratified cross-validation pipeline. The industrial CAPL dataset is enterprise-confidential and is not included.

## Main contributions

- A heterogeneous graph representation organizes 22 CAPL process variables by physical role.
- Mechanistic priors constrain adaptive graph learning while allowing data-driven topology refinement.
- Relation-specific heterogeneous attention models cross-domain process dependencies.
- Dynamic sample weighting emphasizes difficult and sparsely represented boundary-region samples.
- Fold-specific preprocessing and repeated validation reduce leakage and quantify variability.

## Framework

![GAD-Net framework](images/fig7_framework.jpg)

```text
Authorized CAPL production data
        |
        v
Physical admissibility checks
        |
        v
Repeated stratified five-fold partitioning
        |
        v
Training-fold-only standardization and weighting setup
        |
        v
Mechanism-prior graph construction
        |
        v
Prior-constrained adaptive graph learning
        |
        v
Heterogeneous graph attention prediction
        |
        v
Validation-based early stopping and model selection
        |
        v
Evaluation on the held-out test fold
```

The industrial application context is shown below.

![CAPL application scenario](images/fig5_application.jpg)

## Experimental protocol

The default experiment uses five independent random seeds and five stratified folds. Stratification follows the yield-strength distribution. For each fold, input standardization and dynamic sample-weight parameters are fitted using the training portion only. The validation portion is used for early stopping and model selection; the held-out test fold is evaluated after training.

The runner records fold-level predictions, metrics, and training logs, then summarizes mean performance, sample standard deviation, and approximate 95% confidence intervals. Reported metrics include RMSE, MAE, MAPE, R², and boundary-region (tail) MAE.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA driver if GPU training is required.

## Data

The raw CAPL production data are enterprise-confidential and are not distributed. To run the code, place an authorized local copy at `data/CAPL.csv`, or pass another local path with `--data`.

Do not commit the dataset, predictions, checkpoints, logs, or private industrial metadata to a public repository.

## Running the experiment

From the repository root:

```bash
python main.py --data data/CAPL.csv --gpu 0 --output results_5fold_cv/main
```

Use `--gpu -1` for a CPU smoke test.

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

If you use GAD-Net, please cite the accompanying manuscript:

> Knowledge Guided Heterogeneous Graph Attention Network for Yield Strength Prediction of Cold Rolled Strip Steel in Continuous Annealing Production Lines.

## License

This code is released for research use. The enterprise dataset is not included.
