# PicoGPT.jl — Handoff

## Status: 13/22 issues closed, 19/19 tests passing

## What's done (Layers 0–5)
- **config.jl** — GPTConfig + TrainConfig structs
- **gradcheck.jl** — finite-difference gradient checker
- **tensor_ops.jl** — softmax, layer_norm, relu, cross_entropy (all fwd+bwd), embedding, sinusoidal PE, weight init (Xavier + GPT-2)
- **attention.jl** — linear projections, causal mask, single-head + multi-head attention (fwd+bwd)
- **data.jl** — Shakespeare download, char tokenization, batching, train/val split
- **tests.jl** — harness + Tier 1 (12 tests) + Tier 2 (7 tests)

## What's next (in dependency order)
1. **pico-mrg** — Write Tier 3 tests in tests.jl (model shape tests + overfit test)
2. **pico-bsp** — model.jl: transformer block forward/backward
3. **pico-6c6** — model.jl: full GPT model forward/backward + init
4. **pico-mq9** — Write Tier 5 tests (SGD step + LR schedule + gradient clipping)
5. **pico-otd** — Write Tier 6 tests (generation determinism + temperature + top-k)
6. **pico-963** — train.jl: SGD, LR schedule, gradient clipping
7. **pico-54g** — train.jl: training loop with eval
8. **pico-0uk** — generate.jl: autoregressive generation
9. **pico-0c0** — run.jl: entry point

## Key architectural notes
- All arrays use Julia column-major layout: (D, T, B)
- Attention stores per-row softmax caches for backward pass
- The `copy_attention_params` helper exists for gradcheck (mutate a copy)
- Tier 4 data tests (pico-b8h) were closed with placeholder — need filling when data.jl is tested
- The **overfit-one-batch test** (Tier 3) is the critical gate — if a small model can't memorize 4 sequences in 200 steps, there's a backward bug

## Run tests
```
julia tests.jl
```
