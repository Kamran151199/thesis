# `src/training/` — the fine-tuning loop

## Mental model

The `Trainer` is backbone/dataset/objective-agnostic: it pulls batches, asks the
**objective** for a loss, and fires **callbacks** for everything else (logging,
wandb, checkpoints). To change behavior, add a callback — don't touch the loop.

```
on_train_begin
  └─ per optimizer step → on_step_end(step, lr, components)
  └─ per eval           → on_evaluate(step, metrics)
on_train_end(summary)
```

## Gradient accumulation (batch-4 → effective-batch-32)

The A100 fits ~4 examples; the proposal wants batch 32. So we forward/backward 8
micro-batches, summing gradients, then take **one** optimizer step. Each micro
loss is divided by `grad_accum_steps` so the sum equals the true batch-32 mean.
"Step" = optimizer step everywhere (logging, LR schedule, `max_steps`).

## Files

| File | Role |
|------|------|
| `trainer.py` | the loop: accumulation, grad clipping, scheduler step, callback events, mid/final eval |
| `optim.py` | AdamW over trainable params + warmup-cosine scheduler (proposal §7.4) |
| `callbacks.py` | `Callback` base + `ConsoleCallback`, `WandbCallback`, `CheckpointCallback` |
| `checkpoint.py` | save/load **only the trained delta** — LoRA adapter, or the `requires_grad` state dict; writes `meta.json` for provenance |

## What gets saved

Never the frozen 4-bit base (gigabytes, unchanged). Only the delta:
- QLoRA model → `model.save_pretrained()` (a few MB adapter).
- full-tune model → a state dict of `requires_grad` params (the proven pattern).

Plus a `meta.json` (resolved config + final metrics) so a checkpoint folder
explains itself months later.
