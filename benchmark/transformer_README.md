# Transformer Baseline

Pre-LN LLaMA-style transformer (RMSNorm, RoPE, SwiGLU) for char-level benchmarks. Architecture follows [LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling). Configs: `10M`, `5M` (ctx=1024), `10M_ctx20`, `5M_ctx20` (ctx=20).

- **10M-char segments:** `train_text8_transformer.py` — nine runs per config (`--model_no` 1–9), 10M characters each.
- **100M regime (90M chars per run):** `train_text8_transformer_100M.py` — same four model sizes; checkpoints are written as `..._text8_100M.pt`. Text8 is ~100M characters total, so use `--model_no 1` only (one long run per config, not nine folds).
- **text8 eval (100M regime):** `text8_eval_transformer_100M.py` — loads `..._text8_100M.pt`, uses 90M-char segments (`train_sample = 90_000_000`), default `--total_models 1`; writes `../pickle_files/text8_transformer_{size}_res_100M.pickle`.
- **RNN baselines (100M regime):** `train_text8_baselines_100M.py` — use `--cell_type rnn|lstm|gru` for one baseline per process (and `--device cuda:N` per GPU), or `--cell_type all` to run RNN → LSTM → GRU on one device. Checkpoints: `../saved_models/baselines/{rnn,lstm,gru}_model{m}_text8_100M.pt`.
- **text8 RNN baseline eval (100M regime):** `text8_eval_baselines_100M.py` — `--model_type rnn|lstm|gru` for each checkpoint; loads `..._text8_100M.pt`, uses 90M-char segments; default `--total_models 1`; writes `../pickle_files/text8_{rnn|lstm|gru}_res_100M.pickle`.

All commands must be run from the `benchmark/` directory:

```bash
cd /localdisk/ssrivas9/sharp/benchmark
```

## text8 training (9 models, sequential per config)

```bash
mkdir -p logs/text8

# 10M ctx=1024  -- run on cuda:0
for m in 1 2 3 4 5 6 7 8 9; do
    python -u train_text8_transformer.py --model_size 10M --model_no $m --device cuda:0 2>&1 | tee logs/text8/10M_m${m}.log
done

# 5M ctx=1024  -- run on cuda:1
for m in 1 2 3 4 5 6 7 8 9; do
    python -u train_text8_transformer.py --model_size 5M --model_no $m --device cuda:1 2>&1 | tee logs/text8/5M_m${m}.log
done

# 10M ctx=20  -- run on cuda:2
for m in 1 2 3 4 5 6 7 8 9; do
    python -u train_text8_transformer.py --model_size 10M_ctx20 --model_no $m --device cuda:2 2>&1 | tee logs/text8/10M_ctx20_m${m}.log
done

# 5M ctx=20  -- run on cuda:3
for m in 1 2 3 4 5 6 7 8 9; do
    python -u train_text8_transformer.py --model_size 5M_ctx20 --model_no $m --device cuda:3 2>&1 | tee logs/text8/5M_ctx20_m${m}.log
done
```

To run all four configs in parallel across GPUs, background each loop:

```bash
mkdir -p logs/text8

for m in 1 2 3 4 5 6 7 8 9; do python -u train_text8_transformer.py --model_size 10M       --model_no $m --device cuda:0 2>&1 | tee logs/text8/10M_m${m}.log;       done &
for m in 1 2 3 4 5 6 7 8 9; do python -u train_text8_transformer.py --model_size 5M         --model_no $m --device cuda:1 2>&1 | tee logs/text8/5M_m${m}.log;         done &
for m in 1 2 3 4 5 6 7 8 9; do python -u train_text8_transformer.py --model_size 10M_ctx20  --model_no $m --device cuda:2 2>&1 | tee logs/text8/10M_ctx20_m${m}.log;  done &
for m in 1 2 3 4 5 6 7 8 9; do python -u train_text8_transformer.py --model_size 5M_ctx20   --model_no $m --device cuda:3 2>&1 | tee logs/text8/5M_ctx20_m${m}.log;   done &
```

## text8 training — 100M regime (four configs, `model_no=1`)

Uses `train_text8_transformer_100M.py` (90M training characters per run, default `--device cuda`). Pick GPU ids to match your machine.

```bash
mkdir -p logs/text8_100M

# 10M params, ctx=1024
python -u train_text8_transformer_100M.py --model_size 10M --model_no 1 --device cuda:0 \
  2>&1 | tee logs/text8_100M/10M_m1.log

# 5M params, ctx=1024
python -u train_text8_transformer_100M.py --model_size 5M --model_no 1 --device cuda:1 \
  2>&1 | tee logs/text8_100M/5M_m1.log

# 10M params, ctx=20
python -u train_text8_transformer_100M.py --model_size 10M_ctx20 --model_no 1 --device cuda:2 \
  2>&1 | tee logs/text8_100M/10M_ctx20_m1.log

# 5M params, ctx=20
python -u train_text8_transformer_100M.py --model_size 5M_ctx20 --model_no 1 --device cuda:3 \
  2>&1 | tee logs/text8_100M/5M_ctx20_m1.log
```

Run all four transformer variants in parallel (background each line):

```bash
mkdir -p logs/text8_100M

python -u train_text8_transformer_100M.py --model_size 10M       --model_no 1 --device cuda:0 2>&1 | tee logs/text8_100M/10M_m1.log       &
python -u train_text8_transformer_100M.py --model_size 5M         --model_no 1 --device cuda:1 2>&1 | tee logs/text8_100M/5M_m1.log         &
python -u train_text8_transformer_100M.py --model_size 10M_ctx20  --model_no 1 --device cuda:2 2>&1 | tee logs/text8_100M/10M_ctx20_m1.log  &
python -u train_text8_transformer_100M.py --model_size 5M_ctx20   --model_no 1 --device cuda:3 2>&1 | tee logs/text8_100M/5M_ctx20_m1.log   &
```

## text8 RNN baselines — 100M regime (`train_text8_baselines_100M.py`)

Use `--model_no 1` for text8 (90M training chars per segment). Pick `--cell_type` and `--device` to match your machine. Logging matches the transformer runs: unbuffered Python plus `tee` to a log file under `logs/text8_baselines_100M/`.

**One baseline per process (separate GPUs):**

```bash
mkdir -p logs/text8_baselines_100M

python -u train_text8_baselines_100M.py --model_no 1 --cell_type rnn  --device cuda:0 \
  2>&1 | tee logs/text8_baselines_100M/rnn_m1.log

python -u train_text8_baselines_100M.py --model_no 1 --cell_type lstm --device cuda:1 \
  2>&1 | tee logs/text8_baselines_100M/lstm_m1.log

python -u train_text8_baselines_100M.py --model_no 1 --cell_type gru  --device cuda:2 \
  2>&1 | tee logs/text8_baselines_100M/gru_m1.log
```

**All three in parallel (background each line; adjust GPU ids):**

```bash
mkdir -p logs/text8_baselines_100M

python -u train_text8_baselines_100M.py --model_no 1 --cell_type rnn  --device cuda:0 2>&1 | tee logs/text8_baselines_100M/rnn_m1.log   &
python -u train_text8_baselines_100M.py --model_no 1 --cell_type lstm --device cuda:1 2>&1 | tee logs/text8_baselines_100M/lstm_m1.log &
python -u train_text8_baselines_100M.py --model_no 1 --cell_type gru  --device cuda:2 2>&1 | tee logs/text8_baselines_100M/gru_m1.log   &
```

**Sequential on one GPU (same behavior as before the split):**

```bash
mkdir -p logs/text8_baselines_100M

python -u train_text8_baselines_100M.py --model_no 1 --cell_type all --device cuda:0 \
  2>&1 | tee logs/text8_baselines_100M/baselines_all_m1.log
```

## text8 evaluation

```bash
mkdir -p logs/text8_eval

python -u text8_eval_transformer.py --model_size 10M       --device cuda:0 2>&1 | tee logs/text8_eval/10M.log &
python -u text8_eval_transformer.py --model_size 5M         --device cuda:1 2>&1 | tee logs/text8_eval/5M.log &
python -u text8_eval_transformer.py --model_size 10M_ctx20  --device cuda:2 2>&1 | tee logs/text8_eval/10M_ctx20.log &
python -u text8_eval_transformer.py --model_size 5M_ctx20   --device cuda:3 2>&1 | tee logs/text8_eval/5M_ctx20.log &
```

## text8 evaluation — 100M regime (`text8_eval_transformer_100M.py`)

Evaluates checkpoints from `train_text8_transformer_100M.py`. Default `--total_models 1` matches a single trained model per config. Same forward / backward / current splits as the 10M eval script, but with `train_sample = 90_000_000` and checkpoint names ending in `_text8_100M.pt`.

```bash
mkdir -p logs/text8_eval_100M

python -u text8_eval_transformer_100M.py --model_size 10M       --device cuda:0 2>&1 | tee logs/text8_eval_100M/10M.log       &
python -u text8_eval_transformer_100M.py --model_size 5M         --device cuda:1 2>&1 | tee logs/text8_eval_100M/5M.log         &
python -u text8_eval_transformer_100M.py --model_size 10M_ctx20  --device cuda:2 2>&1 | tee logs/text8_eval_100M/10M_ctx20.log  &
python -u text8_eval_transformer_100M.py --model_size 5M_ctx20   --device cuda:3 2>&1 | tee logs/text8_eval_100M/5M_ctx20.log   &
```

## text8 RNN baselines evaluation — 100M regime (`text8_eval_baselines_100M.py`)

Evaluates checkpoints from `train_text8_baselines_100M.py` (`{rnn,lstm,gru}_model{m}_text8_100M.pt`). Default `--total_models 1` matches `--model_no 1` training. Pick `--model_type` and `--device` to match your setup; logs use the same `python -u` + `tee` pattern as other eval scripts.

**One baseline per GPU (sequential commands):**

```bash
mkdir -p logs/text8_eval_baselines_100M

python -u text8_eval_baselines_100M.py --model_type rnn  --device cuda:0 \
  2>&1 | tee logs/text8_eval_baselines_100M/rnn.log

python -u text8_eval_baselines_100M.py --model_type lstm --device cuda:1 \
  2>&1 | tee logs/text8_eval_baselines_100M/lstm.log

python -u text8_eval_baselines_100M.py --model_type gru  --device cuda:2 \
  2>&1 | tee logs/text8_eval_baselines_100M/gru.log
```

**All three in parallel (background each line; adjust GPU ids):**

```bash
mkdir -p logs/text8_eval_baselines_100M

python -u text8_eval_baselines_100M.py --model_type rnn  --device cuda:0 2>&1 | tee logs/text8_eval_baselines_100M/rnn.log   &
python -u text8_eval_baselines_100M.py --model_type lstm --device cuda:1 2>&1 | tee logs/text8_eval_baselines_100M/lstm.log &
python -u text8_eval_baselines_100M.py --model_type gru  --device cuda:2 2>&1 | tee logs/text8_eval_baselines_100M/gru.log   &
```

Optional: `--model_dir` (default `../saved_models/baselines`), `--pickle_dir` (default `../pickle_files`), `--total_models` if you trained multiple segments.

## PG-19 training + evaluation

```bash
mkdir -p logs/pg19

python -u train_pg19_transformer.py --model_size 10M       --device cuda:0 2>&1 | tee logs/pg19/10M.log &
python -u train_pg19_transformer.py --model_size 5M         --device cuda:1 2>&1 | tee logs/pg19/5M.log &
python -u train_pg19_transformer.py --model_size 10M_ctx20  --device cuda:2 2>&1 | tee logs/pg19/10M_ctx20.log &
python -u train_pg19_transformer.py --model_size 5M_ctx20   --device cuda:3 2>&1 | tee logs/pg19/5M_ctx20.log &
```
