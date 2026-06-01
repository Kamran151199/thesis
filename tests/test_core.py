"""
Core unit tests that need NO GPU and NO model/dataset downloads.

They cover the logic most likely to harbor bugs — config parsing, the prompt /
span templates, the masked cross-entropy, and the contrastive/retrieval math —
so you can validate the framework on a laptop before burning A100 minutes.

Run::

    uv run python tests/test_core.py        # plain asserts, prints a summary
    uv run python -m pytest tests/          # if pytest is installed
"""

from __future__ import annotations

import torch

from src.config import ExperimentConfig, dump_config, load_config
from src.data.example import VLMExample
from src.data.prompts import build_template
from src.objectives.base import masked_token_ce
from src.objectives.contrastive import info_nce
from src.evaluation.metrics.retrieval import retrieval_recall


def _fake_example(with_expl: bool = True) -> VLMExample:
    return VLMExample(
        image=None,  # not touched by the text-only logic under test
        question="What gas do plants absorb?",
        choices=["Oxygen", "Carbon dioxide", "Nitrogen"],
        answer_index=1,
        answer="Carbon dioxide",
        explanation="Plants take in CO2 for photosynthesis." if with_expl else None,
    )


def test_registry_uniqueness():
    from src.registry import Registry

    reg: Registry[int] = Registry("thing")
    reg.register("a")(int)
    try:
        reg.register("a")(int)
    except KeyError:
        pass
    else:
        raise AssertionError("duplicate registry key should raise")
    assert reg.available() == ["a"]


def test_config_roundtrip(tmp_path="/tmp"):
    cfg = ExperimentConfig(name="t", rq="RQ2")
    path = f"{tmp_path}/_cfg_roundtrip.yaml"
    dump_config(cfg, path)
    loaded = load_config(path)
    assert loaded.name == "t"
    assert loaded.model.lora.r == 16  # nested dataclass parsed
    assert loaded.train.adam_beta1 == 0.9


def test_config_overrides(tmp_path="/tmp"):
    cfg = ExperimentConfig(name="t")
    path = f"{tmp_path}/_cfg_override.yaml"
    dump_config(cfg, path)
    loaded = load_config(path, overrides=["train.lr=1e-4", "objective.alpha=0.0"])
    assert loaded.train.lr == 1e-4
    assert loaded.objective.alpha == 0.0


def test_answer_only_template():
    t = build_template("answer_only")
    ex = _fake_example()
    prompt, target = t(ex)
    assert (
        prompt
        == "Question: What gas do plants absorb? Options: (A) Oxygen, (B) Carbon dioxide, (C) Nitrogen."
    )
    assert target == " Answer: Carbon dioxide."
    assert t.target_spans(ex) == [("answer", " Answer: Carbon dioxide.")]


def test_explanation_template_spans():
    t = build_template("explanation_then_answer")
    ex = _fake_example()
    _, target = t(ex)
    spans = t.target_spans(ex)
    # spans must concatenate back to exactly the target (collator relies on this)
    assert "".join(s for _, s in spans) == target
    assert [name for name, _ in spans] == ["explanation", "answer"]
    # falls back to answer-only when no rationale present
    assert build_template("explanation_then_answer").target_spans(
        _fake_example(False)
    ) == [("answer", " Answer: Carbon dioxide.")]


def test_masked_token_ce_selects_subset():
    torch.manual_seed(0)
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    full = masked_token_ce(logits, labels)  # all positions
    answer_mask = torch.zeros(B, T, dtype=torch.bool)
    answer_mask[:, -2:] = True  # only last 2 positions
    sub = masked_token_ce(logits, labels, mask=answer_mask)
    assert full.item() > 0 and sub.item() > 0
    # empty mask → differentiable zero, not NaN
    empty = masked_token_ce(logits, labels, mask=torch.zeros(B, T, dtype=torch.bool))
    assert empty.item() == 0.0


def test_info_nce_perfect_alignment_is_low():
    emb = torch.eye(4)  # orthogonal pairs
    loss_aligned = info_nce(emb, emb, temperature=0.07)
    shuffled = emb[torch.tensor([1, 2, 3, 0])]
    loss_misaligned = info_nce(emb, shuffled, temperature=0.07)
    assert loss_aligned < loss_misaligned


def test_retrieval_recall_perfect():
    emb = torch.randn(8, 16)
    out = retrieval_recall(emb, emb.clone())  # identical → rank 1 for all
    assert out["R@1"] == 1.0 and out["MRR"] == 1.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\n{len(tests)} core tests passed.")


if __name__ == "__main__":
    _run_all()
