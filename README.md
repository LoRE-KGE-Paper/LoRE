
# LoRE: End-to-End Dynamic Knowledge Graph Embeddings via Local Reconstructions

This repository contains the official supplementary code for our ISWC 2025 submission:

**LoRE: End-to-End Dynamic Knowledge Graph Embeddings via Local Reconstructions**

LoRE is a hybrid approach to dynamic knowledge graph embeddings (KGEs). It extends translational KGE models with local GNN-based reconstructions to enable embedding updates without costly full retraining.

---

## Installation (via Conda)

Set up the environment with:

```bash
conda env create -f environment.yml
conda activate lore
```

This installs all dependencies needed for training, reconstruction, updating, and retraining. The environment is CUDA-ready but also runs on CPU (with increased runtimes).

---

## Usage Examples

### 1. Initial Training

Train base embeddings using the small configuration (CPU-friendly), which restricts subgraph batches to benchmark-labeled nodes. While this speeds up training significantly, some performance degradation should be expected. To reproduce full LoRE results from the paper, use the full configuration (for CPU and MUTAG: still feasible on CPU):

```bash
python train.py --config config/train/aifb-transE-small.json
```

```bash
python train.py --config config/train/aifb-transE.json
```

---

### 2. Embedding Reconstruction

Reconstruct node embeddings (e.g., for downstream classification) from a saved model without retraining:

```bash
python reconstruct.py --config config/reconstruct/aifb-transE.json
```

---

### 3. Dynamic Graph Update

Simulate dynamic updates with node additions and deletions, followed by embedding reconstruction. We provide a sample update configuration for AIFB. For other KGs, configs can be defined analogously.

```bash
python update.py --config config/update/aifb-transE-sample.json
```

---

### 4. Retrain on Updated Graph

Retrain embeddings after dynamic updates using previously saved models and locally reconstructed initializations. We provide a retrain config for AIFB. Similar configs can be created for other KGs.

```bash
python retrain.py --config config/retrain/aifb-transE-sample.json
```

---

## Datasets & Configurations

Supported benchmark knowledge graphs:

- `AIFB`
- `MUTAG`
- `BGS`
- `AM`

Each KG includes:

- **Small training config**: `config/train/<dataset>-transE-small.json`  
- **Full training config**: `config/train/<dataset>-transE.json`  
- **Reconstruction config**: `config/reconstruct/*.json`  

The AIFB KG additionally includes:

- **Update and retrain configs**: `config/update/*.json`, `config/retrain/*.json`

Raw KG files are stored in `data/<dataset>/*.n3`, retrieved from PyTorch Geometric and preprocessed in line with prior related work (see paper for details).

---

## Reproducibility

All LoRE embeddings presented in the paper were generated using this codebase:

- **Static graph embeddings** (Section 5.1)
- **Dynamic graph embeddings** (Section 5.2 and 5.3)

The `out/` directory stores trained models and reconstructed embeddings.

---

## Embedding Export & Downstream Use

Reconstructed embeddings are saved as `.npz` files and can be easily reused in downstream tasks such as classification or clustering.

To load them in Python:

```python
import numpy as np

loaded_data = np.load("reconstructions.npz", allow_pickle=True)
loaded_embedding_numpy = loaded_data["embeddings"]
loaded_identifiers = loaded_data["identifiers"].tolist()
```

- `embeddings` contains a NumPy array of shape `(num_nodes, embedding_dim)`
- `identifiers` is a list of node URIs (or node IDs) corresponding to each row in the embedding matrix

This format allows seamless integration with scikit-learn, PyTorch, or other ML frameworks.

---

## Reviewer Notes

- This **command-line-based repository** was used for all LoRE training and evaluation runs.
- Configs allow both lightweight CPU-based and full-scale GPU-based experiments.
- A **Flask-based dynamic API** using the same backend is under active development and will be released upon paper acceptance.

---
