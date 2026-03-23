# Neural FCA — Interpretable Classification via Formal Concept Analysis

Course assignment for **"Ordered Sets in Data Analysis"** at HSE (Higher School of Economics).

## Overview

This project implements an interpretable machine learning classifier by combining **Formal Concept Analysis (FCA)** with neural networks. Instead of a black-box model, the network architecture is derived directly from a concept lattice — making each neuron correspond to a meaningful formal concept and allowing human-readable explanations of predictions.

The approach is applied to a binary classification task: predicting whether a customer will accept a marketing offer.

## How It Works

1. **Binarize** the input features to prepare them for FCA
2. **Build a concept lattice** from the binarized dataset using `fcapy`
3. **Select the best concepts** from the lattice
4. **Construct a neural network** whose layers mirror the structure of the concept poset
5. **Train and evaluate** the model, comparing it against classical ML baselines

The resulting network is sparse, theory-grounded, and traceable — you can inspect which neurons fire for any given input.

## Architecture

The `ConceptNetwork` class (in `neural_lib58.py`) is the core component:

- Input neurons correspond to individual attributes
- Hidden neurons correspond to formal concepts from the lattice
- Connections follow concept ordering relationships from the poset
- Output layer uses Softmax for classification

```python
from neural_lib58 import ConceptNetwork

network = ConceptNetwork.from_lattice(
    lattice=concept_lattice,
    best_concepts_indices=[...],
    targets=('yes', 'no')
)

network.fit(X_df, y_series, n_epochs=2000)

predictions = network.predict(X_test)
probabilities = network.predict_proba(X_test)
```

To trace which concepts activated for a specific sample:

```python
network.trace_description(x_sample)
```

## Dataset

**Superstore Marketing Campaign** (Kaggle) — 2,240 customer records, 22 features.

Target: whether the customer accepted a gold membership offer (binary, imbalanced).

Evaluation metric: **F1 Score** (due to class imbalance).

## Baselines

The neural FCA model is compared against:

- K-Nearest Neighbors
- Naive Bayes
- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- XGBoost

## Project Structure

```
neural-fca/
├── neural_lib58.py                  # ConceptNetwork implementation
├── big_hw_syimyk_khamdamov.ipynb    # Experiments, analysis, and results
├── LICENSE
└── README.md
```

## Dependencies

```
torch
fcapy
SparseLinear
scikit-learn
pandas
```

Install manually or via your preferred environment manager.

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 2000 |
| Loss | CrossEntropyLoss |

## License

MIT — see [LICENSE](LICENSE).
