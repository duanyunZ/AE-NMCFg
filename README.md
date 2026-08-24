# AE-NMCFg

**AE-NMCFg: A Dual-Constrained Nonnegative Matrix Co-Factorization for Student Cognitive Diagnosis**

This code implements the AE-NMCFg model for student cognitive diagnosis, which jointly performs performance prediction and cognitive diagnosis under two pedagogical constraints: (1) psychometric monotonicity, and (2) the Hierarchical Cognitive Assumption (HCA).

---

## Directory Structure

```
project_root/
├── AE_NMCFg/
│   ├── ae_nmcfg.py              # Main model class
│   ├── ae_nmcfg_lipschitz.py    # Core optimization routines (PG-BCD + Adam)
│   ├── main.py                  # Training and evaluation pipeline
│   ├── data/                    # Prerequisite graphs and saved train/test splits
│   └── train_log.csv            # Experiment logs (auto-generated)
├── SidePackage/
│   ├── preprocessing.py         # Data preprocessing utilities
│   ├── auxiliary.py             # Helper functions
│   └── evaluation.py            # Evaluation metrics (ACC, RMSE, KRC, PR)
└── Data/
    └── <dataset_name>/
        ├── data.txt             # Response matrix (students x exercises)
        ├── q.txt                # Q-matrix (exercises x knowledge concepts)
        └── problemdesc.txt      # Problem descriptions (optional)
```

---

## Dataset Preparation

Each dataset must be placed under `Data/<dataset_name>/` with the following files:

1. **data.txt**: Students in rows, exercises in columns. Entries: 1 (correct), 0 (incorrect), or NaN (missing). The code transposes this internally to exercises x students.

2. **q.txt**: Exercises in rows, knowledge concepts in columns. Binary entries: 1 if exercise covers the concept, 0 otherwise.

3. **problemdesc.txt** (optional): One line per exercise. `Obj` for objective questions, `Sub` for subjective questions.

**Prerequisite Graph** (for datasets with HCA constraints):
- Place adjacency matrix at: `AE_NMCFg/data/hier_matrix@<dataset>.txt`
- Format: `adj[i, j] = 1` means concept `i -> j` (i is prerequisite of j)

---

## Running the Code

```bash
cd /path/to/project_root
python AE_NMCFg/main.py
```

Select a dataset from the prompt.

The script will:
- Split data into 80% training / 20% testing
- Train the model using PG-BCD for matrix factors + Adam for GAT
- Evaluate on the test set (ACC, RMSE, KRC, PR_0.05, PR_0.1)
- Save results to `train_log.csv`

---

## Key Hyperparameters (adjust in main.py)

| Parameter | Description | Default |
|-----------|-------------|---------|
| MISS_R | Train/test split missing rate | 0.2 |
| RANK | Number of latent topics T (dataset-specific) | 4 |
| lambda_reg | HCA regularization weight (dataset-specific) | 0.06 |
| beta_reg | GAT reconstruction loss weight (dataset-specific) | 0.005 |
| MAX_ITER | Maximum outer iterations | 500 |
| CRI | Convergence threshold | 1.0 |

---

## Output Metrics

| Metric | Description |
|--------|-------------|
| ACC | Accuracy of response prediction (higher is better) |
| RMSE | Root Mean Square Error (lower is better) |
| KRC | Knowledge-Response Consistency (higher is better) |
| PR_0.05 | Prerequisite Satisfaction Rate at alpha = 0.05 (higher is better) |
| PR_0.1 | Prerequisite Satisfaction Rate at alpha = 0.1 (higher is better) |

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- NumPy
- SciPy

Install dependencies:
```bash
pip install torch torch-geometric numpy scipy pandas
```

---
