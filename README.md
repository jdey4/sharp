# Long Range Non-Stationary Temporal Pattern Recognition via Hierarchical Accelerated Replay

This repository contains the implementation of a hierarchical memory architecture for **single-pass streaming sequence learning** under non-stationary dynamics.

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

This work instead decomposes learning into:Memory (no credit assignment) and Pattern Recognition (with credit assignment)


Memory is:
- accumulated online (wake phase)
- consolidated offline (sleep phase)

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


