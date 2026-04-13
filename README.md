# SHARP: Sleep-based Hierarchical Accelerated Replay for Long Range Non-Stationary Temporal Pattern Recognition

This repository contains the implementation of a hierarchical memory architecture for **single-pass streaming sequence learning** under non-stationary dynamics.

---

## Introduction

The model is inspired by biological memory systems and introduces a separation between:
- **Memory (non-credit-assigned accumulation)**
- **Pattern recognition (credit-assigned inference)**

It leverages **accelerated sequential replay during sleep phases** to extend effective temporal context without increasing online computational cost.

---

## Key Features

- **Single-pass (online) learning** — no revisiting past data  
- **Hierarchical memory organization**  
- **Accelerated sequential replay (sleep phase)**  
- **Adaptive compute allocation** (more compute early, less later)  
- **Improved retention and generalization**
  - Backward BPC → past retention  
  - Current BPC → adaptation  
  - Forward BPC → generalization  

---

## Core Idea

Traditional models conflate memory with weight updates via gradient descent.

This work instead decomposes learning into:

- **Memory (no credit assignment)**  
- **Pattern Recognition (with credit assignment)**  

Memory is:
- accumulated online (**wake phase**)  
- consolidated offline (**sleep phase**)  

During sleep:
- temporally structured sequences are replayed  
- replay is **accelerated via downsampling**  
- higher layers learn long-range structure  

---

## Benchmarks

Evaluated on:
- `text8`  
- `PG-19`  

Metrics:
- **Forward BPC** → future generalization  
- **Backward BPC** → retention of past data  
- **Current BPC** → adaptation to recent data  

---

## Hardware Requirements

The model is lightweight compared to large transformer-based systems, but dataset size (especially PG-19) can be demanding.

### Minimum
- CPU: 4+ cores  
- RAM: 8 GB  
- Storage: ~10–20 GB (datasets + checkpoints)  

### Recommended
- CPU: 8–16 cores  
- RAM: 16–32 GB  
- GPU: Optional (Apple Silicon / CUDA GPU supported via PyTorch)  
- Storage: 50+ GB (for large-scale experiments)

### Notes
- Training is **sequential and streaming**, so GPU is not strictly required  
- Larger memory layers and longer sequences benefit from more RAM  
- PG-19 preprocessing can be slow and storage-heavy  

---

## Setup Instructions

### 1. Change directory to SHARP

```bash
cd sharp
```

### 2. Create environment (recommended)

**Using conda**
```bash
conda create -n sleep python=3.12
conda activate sleep
```

**Or using venv**
```bash
python -m venv sleep_env
source sleep_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package (editable mode)

```bash
pip install -e .
```

> Avoid using `python setup.py install` — it may create dependency conflicts.

### 5. Prepare datasets

Datasets are automatically downloaded or processed during training scripts.

For PG-19:
- First run may take time due to download + preprocessing  
- Ensure sufficient disk space  

### 6. Run training

```bash
python train_pg19.py
```

---

## Repository Structure

```text
sleep_experiment/
│
├── sharp/            # Core model implementation
├── benchmark/        # Benchmark experiment scripts
├── experiments/      # Simulation experiment scripts
├── dataset/          # Data storage (ignored by git)
├── pickle_files/     # Intermediate results (ignored by git)
├── plots/            # Visualization outputs
├── saved_models/     # Checkpoints (ignored by git)
└── requirements.txt  # Dependencies
```

---

## Notes

- The system is designed for **single-pass streaming learning**
- Performance depends heavily on:
  - memory hierarchy depth  
  - replay schedule  
  - downsampling rate  