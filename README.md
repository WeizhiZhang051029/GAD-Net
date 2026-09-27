# 🧠 GAD-Net: Knowledge-Guided Adaptive Graph Network for Steel Yield-Strength Prediction

<p align="center">
<b>
Knowledge-Guided Graph Learning · Adaptive Topology Refinement · Heterogeneous Graph Attention · Tail-aware Adaptive Optimization
</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/GAD--Net-Yield%20Strength%20Prediction-blue">
<img src="https://img.shields.io/badge/Knowledge--Guided-Graph%20Learning-green">
<img src="https://img.shields.io/badge/Adaptive-Topology%20Refinement-purple">
<img src="https://img.shields.io/badge/Heterogeneous-Graph%20Attention-orange">
</p>


<p align="center">
  <b>Official implementation of GAD-Net for steel yield-strength prediction in continuous annealing production lines.</b>
</p>


<p align="center">
  Project Page | Paper
</p>


---

# 📌 Overview

This repository provides the official implementation of **GAD-Net: Knowledge-Guided Adaptive Graph Network for Steel Yield-Strength Prediction**.

Yield strength is a critical quality indicator in continuous annealing production lines (CAPLs). Accurate prediction remains challenging due to limited industrial samples, heterogeneous process variables, strong variable coupling, and insufficient representation of underlying process mechanisms.

Existing data-driven methods mainly rely on statistical correlations and often ignore physical process knowledge. Meanwhile, conventional graph-based approaches usually employ fixed graph structures, limiting their ability to capture complex and evolving relationships among manufacturing variables.

To address these challenges, **GAD-Net** integrates mechanism-prior graph construction, adaptive topology refinement, heterogeneous graph attention learning, and tail-aware adaptive sample weighting into a unified framework.


The GAD-Net framework consists of four main components:

* **Mechanism-prior graph construction:** Incorporates CAPL process knowledge into graph initialization and provides physically meaningful structural constraints.

* **Adaptive topology refinement:** Learns complementary data-driven variable relationships beyond predefined physical connections.

* **Heterogeneous graph attention learning:** Captures complex interactions among different categories of process variables.

* **Tail-aware adaptive weighting:** Improves prediction robustness in sparse and difficult target regions.


The public release contains the complete GAD-Net implementation and cross-validation experimental pipeline. The industrial CAPL dataset is enterprise-confidential and therefore not included.

<p align="center">
<img src="images/fig4_workflow.jpg" width="100%">
</p>

<p align="center">
<em>Overall workflow of GAD-Net for steel yield-strength prediction in continuous annealing production lines.</em>
</p>


---

# 🔥 Highlights

* A knowledge-guided adaptive graph network is proposed for steel yield-strength prediction.

* Physical process knowledge is embedded into graph construction to improve industrial interpretability.

* Adaptive topology refinement discovers hidden variable interactions from production data.

* Heterogeneous graph attention models complex dependencies among different process variable groups.

* Tail-aware adaptive weighting improves prediction performance in challenging target regions.

* Leakage-safe preprocessing ensures reliable industrial evaluation.

* Complete cross-validation experiments are provided for reproducible research.


---

# 🧩 Framework


<p align="center">
<img src="images/fig7_framework.jpg" width="100%">
</p>


The complete training workflow of GAD-Net is organized as follows:


Authorized CAPL production data

        |
        v

Physical admissibility filtering

        |
        v

Stratified cross-validation partition

        |
        v

Training-fold-only preprocessing

        |
        v

Mechanism-prior graph construction

        |
        v

Adaptive topology refinement

        |
        v

Heterogeneous graph attention learning

        |
        v

Tail-aware adaptive optimization

        |
        v

Validation-based model selection

        |
        v

Final yield-strength prediction


<p align="center">
<img src="images/fig5_application.jpg" width="100%">
</p>

<p align="center">
<em>Industrial application scenario of GAD-Net in continuous annealing production lines.</em>
</p>


---

# ⚙️ Method Overview


## 1. Knowledge-Guided Graph Construction

GAD-Net first constructs a mechanism-prior graph based on domain knowledge from the continuous annealing process.

The process variables are organized into four heterogeneous node groups:

* Operating variables

* Procedure variables

* Conditional variables

* Composition variables


The initial graph topology provides physically meaningful structural constraints while enabling further graph learning.


---

## 2. Adaptive Topology Refinement

Although physical knowledge provides valuable prior information, industrial processes contain complex hidden dependencies.

GAD-Net introduces adaptive topology refinement to update graph structures according to feature relationships, allowing the model to discover complementary data-driven interactions while maintaining physical consistency.


---

## 3. Heterogeneous Graph Attention Learning

A heterogeneous graph attention module is designed to capture interactions among different variable categories.

Through multi-head attention aggregation, GAD-Net learns:

* Cross-category variable dependencies.

* High-order process interactions.

* Condition-dependent feature representations.


---

## 4. Tail-aware Adaptive Weighting

Industrial yield-strength data often exhibit uneven target distributions.

GAD-Net introduces tail-aware adaptive weighting to emphasize difficult samples and improve prediction robustness in sparse target regions.


---

# 📊 Experimental Protocol

The default experiment evaluates GAD-Net using stratified cross-validation.


The experimental protocol follows strict leakage prevention principles:

* Dataset partitioning is performed before model training.

* Feature statistics and preprocessing parameters are fitted only using the corresponding training portion.

* Sample-weight parameters are estimated only from training data.

* Validation data are used for early stopping and model selection.

* Test data are evaluated only after training completion.


Multiple independent experiments with fixed random seeds are conducted to evaluate model stability and reliability.


The runner records individual experimental results and summarizes the overall performance statistics, including average performance, variability, and confidence estimation.


---

# ⚙️ Configuration


Representative experimental settings:


| Component | Setting |
|------------|------------|
| Task | Steel yield-strength prediction |
| Data type | Industrial CAPL production records |
| Validation strategy | Stratified cross-validation |
| Graph type | Knowledge-guided heterogeneous graph |
| Graph refinement | Adaptive topology learning |
| Attention mechanism | Multi-head heterogeneous graph attention |
| Optimization | Tail-aware adaptive training |
| Evaluation | Regression metrics |


Detailed architecture and optimization parameters are provided in the configuration files.


---

# 🛠️ Installation


Clone the repository:



git clone https://github.com/your_username/GAD-Net.git

cd GAD-Net



Create and activate a virtual environment:



python -m venv .venv

source .venv/bin/activate



Install dependencies:


python -m pip install --upgrade pip

pip install -r requirements.txt



Install a PyTorch version compatible with your CUDA environment if GPU acceleration is required.


---


# 🚀 Running the Main Experiment


Run the complete experiment:


```bash
python main.py \
    --data data/CAPL.csv \
    --gpu 0 \
    --output results/main
```


For CPU smoke testing:


```bash
python main.py \
    --data data/CAPL.csv \
    --gpu -1
```


The pipeline sequentially performs:


1. Data loading and preprocessing.

2. Cross-validation partitioning.

3. Knowledge-guided graph construction.

4. Adaptive graph learning.

5. Model optimization.

6. Validation-based model selection.

7. Final evaluation.


---

# 📁 Repository Structure


```text
GAD-Net/

├── images/
│   ├── fig4_workflow.jpg
│   ├── fig5_application.jpg
│   └── fig7_framework.jpg
│
├── models/
│   ├── gad_net.py
│   └── __init__.py
│
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


The main components are organized as follows:


* `models/`

  Implementation of GAD-Net architecture, including graph learning and attention modules.


* `graph_construction.py`

  Construction of mechanism-prior graph topology and adaptive graph initialization.


* `data_preprocessing.py`

  Dataset loading and preprocessing utilities.


* `fold_preprocessing.py`

  Leakage-safe fold-specific preprocessing.


* `train.py`

  Model optimization and training procedures.


* `evaluate.py`

  Prediction evaluation and metric calculation.


* `main.py`

  Complete experimental pipeline.


---

# 📦 Outputs


Experimental results are stored under:


```text
results/
```


Typical outputs:


```text
results/

├── checkpoints/

├── predictions/

├── metrics/

├── logs/

└── summary.json
```


The output files contain:


* Model checkpoints.

* Prediction results.

* Evaluation metrics.

* Training logs.

* Experiment summaries.


---

# 📏 Evaluation Metrics


GAD-Net is evaluated using four regression metrics:


* **Root Mean Squared Error (RMSE)**

* **Mean Absolute Error (MAE)**

* **Mean Absolute Percentage Error (MAPE)**

* **Coefficient of Determination (R²)**


Lower RMSE, MAE, and MAPE values indicate smaller prediction errors, while higher R² indicates stronger agreement between predicted and measured yield strength.


---

# 📰 News


* GAD-Net framework completed.

* Experimental validation completed.

* Source code released.


---

# 🙏 Acknowledgements


This project builds upon advances in:

* Graph neural networks.

* Heterogeneous graph learning.

* Knowledge-guided machine learning.

* Industrial artificial intelligence.


We thank the open-source community for providing valuable tools and resources.


---

# 📖 Citation


If you find GAD-Net useful in your research, please consider citing our paper:

```text
@article{gadnet2026,
  title={GAD-Net: Knowledge-Guided Adaptive Graph Network for Steel Yield-Strength Prediction},
  year={2026}
}
```


Citation information will be updated after the paper is officially published.


---

# 📬 Contact


For questions regarding implementation, experimental configuration, or reproducibility, please open an issue in this repository.
