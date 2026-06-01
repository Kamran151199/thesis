# Cross-Modal Generative Alignment for Document, Chart, and Image Reasoning with Explanation-Aware VLMs

Bachelor's thesis code. A registry-driven, config-first framework for fine-tuning
vision-language models (VLMs) under different alignment objectives and measuring
their reasoning accuracy, explanation quality, and faithfulness.

> Target venue: *Engineering Applications of Artificial Intelligence (EAAI)*.
> Compute: 1× NVIDIA A100-40GB (Colab Pro+). Training: **QLoRA only**.

## Research questions (what every experiment serves)

1. **RQ1** how VLM alignment evolved (survey)
2. **RQ2** generative vs contrastive alignment
3. **RQ3** does *explanation-aware* training help? — the core contribution
4. **RQ4** architectural/training factors across domains
5. **RQ5** faithfulness & hallucination of explanations
6. **RQ6** QLoRA across scales (2B vs 7B)
7. **RQ7** cross-domain transfer

## Layout

```
src/            the experiment framework  ── see src/README.md for the architecture
  config/   data/   models/   objectives/   training/   evaluation/   experiment/
configs/        one YAML per experiment    ── see configs/README.md
exploration/    from-scratch learning files (ViT, CLIP, mini-LLaVA, …)
colab/          Colab bootstrap + the original BLIP-2 prototype
tests/          CPU unit tests (no GPU / no downloads)
thesis_doc/     the written thesis + proposal
notes/          paper notes, video transcripts
```

## Quick start

```bash
uv sync                                   # install (editable: `import src` works)
uv run python tests/test_core.py          # sanity-check the framework (CPU)
uv run python -m src.run --list           # show registered backbones/datasets/objectives/metrics

# run an experiment (needs a GPU for the real models)
uv run python -m src.run --config configs/experiments/rq2_blip2_generative_scienceqa.yaml
```

## On Colab

`colab/setup_colab.py` installs this package editable on the A100 kernel and
points the HuggingFace cache at Drive. Then it's the same CLI:

```python
exec(open("colab/setup_colab.py").read())   # one-time per session
!python -m src.run --config configs/experiments/rq3_qwen2vl_explanation_aware_scienceqa.yaml
```

## The model

Four pluggable axes — pick one of each in a YAML file:

```
backbone (blip2 · qwen2_vl · paligemma)   ×   dataset (scienceqa · chartqa · docvqa · aokvqa · vqav2)
objective (generative · explanation_aware · contrastive)   ×   metric (mc_accuracy · anls · rouge_l · …)
```

Adding any option = one new file + one `@REGISTRY.register("name")` decorator.
See [`src/README.md`](src/README.md) for the full architecture and data flow.

## License

Apache 2.0 (per the proposal).
