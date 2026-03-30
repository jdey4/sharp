# Transformer Baseline

Pre-LN LLaMA-style transformer (RMSNorm, RoPE, SwiGLU) for char-level benchmarks. Architecture follows [LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling). Configs: `10M`, `5M` (ctx=1024), `10M_ctx20`, `5M_ctx20` (ctx=20).

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

## text8 evaluation

```bash
mkdir -p logs/text8_eval

python -u text8_eval_transformer.py --model_size 10M       --device cuda:0 2>&1 | tee logs/text8_eval/10M.log &
python -u text8_eval_transformer.py --model_size 5M         --device cuda:1 2>&1 | tee logs/text8_eval/5M.log &
python -u text8_eval_transformer.py --model_size 10M_ctx20  --device cuda:2 2>&1 | tee logs/text8_eval/10M_ctx20.log &
python -u text8_eval_transformer.py --model_size 5M_ctx20   --device cuda:3 2>&1 | tee logs/text8_eval/5M_ctx20.log &
```

## PG-19 training + evaluation

```bash
mkdir -p logs/pg19

python -u train_pg19_transformer.py --model_size 10M       --device cuda:0 2>&1 | tee logs/pg19/10M.log &
python -u train_pg19_transformer.py --model_size 5M         --device cuda:1 2>&1 | tee logs/pg19/5M.log &
python -u train_pg19_transformer.py --model_size 10M_ctx20  --device cuda:2 2>&1 | tee logs/pg19/10M_ctx20.log &
python -u train_pg19_transformer.py --model_size 5M_ctx20   --device cuda:3 2>&1 | tee logs/pg19/5M_ctx20.log &
```
