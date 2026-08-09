# Neural FCA — Interpretable Classification via Formal Concept Analysis

A classifier whose hidden layer *is* a concept lattice. Every neuron is a formal concept with a
readable intent, so a prediction traces back to the exact conjunction of attributes that caused
it — no post-hoc explainer involved, the explanation *is* the architecture.

Research project for **Ordered Sets in Data Analysis**, HSE Faculty of Computer Science.

---

## Results

Superstore Marketing Campaign — predict which existing customers accept a gold-membership offer.
2,227 customers after cleaning, 15% positive class, stratified 80/20 split, F1 on the positive
class (accuracy is uninformative at a 5.7:1 imbalance).

| Model | F1 (test) | Notes |
|---|---|---|
| Random Forest | **0.578** | Optuna, 5-fold CV (CV-F1 0.533) |
| XGBoost | 0.578 | Optuna, 5-fold CV (CV-F1 0.557) |
| Logistic Regression | 0.528 | Optuna, 5-fold CV (CV-F1 0.482) |
| **Neural FCA** | **0.396** | 15 concepts, oversampled |
| kNN | 0.295 | Optuna, 5-fold CV (CV-F1 0.347) |
| Neural FCA — no oversampling | 0.000 | ablation, see below |

**Neural FCA beats kNN and trails the tuned tree ensembles by ~0.18 F1.** That gap is the price
of interpretability on this dataset, and it is driven by three deliberate constraints rather
than by the method itself:

- **The lattice is built on 60 sampled objects.** Concept lattice construction is exponential in
  the worst case — 60 rows over 10 attributes already produce 334 concepts — so the network
  selects concepts from a small sample while the baselines see all 1,781 training rows.
- **Ten binary attributes, selected by marginal correlation.** Quantile binning into 3 bins plus
  a correlation filter discards interactions that are invisible marginally.
- **The network is sparse by construction.** Edges exist only where the concept order relation
  puts them, so it cannot fit arbitrary feature interactions the way a boosted ensemble can.

### Ablation: class balancing is load-bearing

Removing `RandomOverSampler` and changing nothing else, the network converges to predicting the
majority class for all 446 test customers — **F1 = 0.000**. Cross-entropy on a 5.7:1 split is
minimized well enough by "always say no" that the concept layer never has to separate anything,
and the interpretable structure collapses to a constant. For Neural FCA the balancing step is
not a tuning detail; it is what makes the model learn at all.

### What the model buys

For any customer the set of activated concepts is a readable conjunction. The strongest single
concept on this dataset is `Tenure_years_high ∧ Recency_low` (F1 = 0.677 measured on the 60-object subset the lattice is built from) — long-term
customers who bought recently. `trace_description()` returns the active concepts for one sample
and the notebook renders them on the lattice diagram.

---

## Method

1. **Binarize** — quantile `KBinsDiscretizer` (3 bins) on numeric features, one-hot on
   categoricals → 64 binary attributes.
2. **Select attributes** — top 10 by `|corr|` with the target, to keep the lattice tractable.
3. **Build the lattice** — `fcapy.ConceptLattice.from_context(..., is_monotone=True)` on a
   60-object sample → 334 concepts.
4. **Select concepts** — score every concept by the F1 of using its extent as a prediction, keep
   the top 15.
5. **Build the network** — input neurons = attributes, hidden neurons = selected concepts, edges
   = concept order relation, output = softmax over classes.
6. **Train** — Adam, lr 1e-3, 2,000 epochs, cross-entropy, on the oversampled training set.

The `ConceptNetwork` class lives in [`neural_lib58.py`](neural_lib58.py):

```python
import neural_lib58 as nl

network = nl.ConceptNetwork.from_lattice(
    lattice=concept_lattice,
    best_concepts_indices=best_concept_idx,
    targets=(0, 1),
)
network.fit(X_train_fca, y_train_bal, n_epochs=2000)

network.predict(X_test_fca)
network.predict_proba(X_test_fca)
network.trace_description(frozenset(active_attributes))   # which concepts fired
```

---

## Reproducing

```bash
git clone https://github.com/semthedev/neural-fca.git
cd neural-fca

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab neural_fca_experiments.ipynb
```

Or open it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/semthedev/neural-fca/blob/main/neural_fca_experiments.ipynb)

The notebook fetches the dataset itself via `kagglehub` (credentials from `~/.kaggle/kaggle.json`
or `KAGGLE_USERNAME` / `KAGGLE_KEY`). If you would rather skip Kaggle setup, download
[`superstore_data.csv`](https://www.kaggle.com/datasets/ahsan81/superstore-marketing-campaign-dataset)
manually into `data/` — the download cell detects it and skips.

Splits and samplers are seeded (`random_state=42`); the Optuna studies use `TPESampler(seed=42)`.
Torch weight initialization is not separately seeded, so the Neural FCA number varies slightly between runs.

Requires Python 3.10+.

---

## Layout

```
neural-fca/
├── neural_lib58.py                  # ConceptNetwork: lattice → sparse torch module
├── neural_fca_experiments.ipynb     # full pipeline, baselines, ablation, analysis
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Limitations and next steps

- The 60-object lattice sample is the main bottleneck. A larger subset with a **concept
  stability** filter — rather than ranking concepts by raw F1, which overfits the sample —
  is the obvious next experiment.
- Attribute selection by marginal correlation misses interactions; mutual information or a tree
  ensemble's feature importances would be a stronger filter.
- The comparison is missing the fair interpretable baseline: a **decision tree of matching
  depth**. Against black-box ensembles the interpretability argument is assumed rather than
  tested.

---

## License

MIT — see [LICENSE](LICENSE).
