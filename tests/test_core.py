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

from pathlib import Path

import torch
from PIL import Image

from src.config import ExperimentConfig, dump_config, load_config
from src.data.example import VLMExample
from src.data.prompts import build_template
from src.evaluation.faithfulness import mask_region
from src.objectives.base import masked_token_ce
from src.objectives.contrastive import info_nce
from src.objectives.explanation_aware import (
    effective_answer_alpha,
    supervised_span_counts,
)
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


def test_answer_only_template_excludes_rationale_text():
    t = build_template("answer_only")
    _, target = t(_fake_example())
    assert "Reasoning:" not in target
    assert "photosynthesis" not in target


def test_rationale_template_includes_rationale_then_answer():
    t = build_template("explanation_then_answer")
    _, target = t(_fake_example())
    assert target.index("Reasoning:") < target.index("Answer:")
    assert "photosynthesis" in target


def test_paper_pipeline_transfer_prompt_modes_are_explicit():
    text = Path("notebooks/00_paper_pipeline.py").read_text()
    transfer_section = text.split("TRANSFER_SOURCES = {", 1)[1].split("TRANSFER_TARGETS = {", 1)[0]
    answer_only_sources = [
        "science_answer_only",
        "aokvqa_answer_only",
        "charts_answer_only",
        "documents_answer_only",
        "vqav2_answer_only",
    ]
    rationale_sources = [
        "science_rationale_ce",
        "science_expl_aware",
        "aokvqa_rationale_ce",
        "aokvqa_expl_aware",
    ]
    for source in answer_only_sources:
        block = transfer_section.split(f'"{source}"', 1)[1].split("},", 1)[0]
        assert '"prompt_variant": "answer_only"' in block
        assert '"cot": False' in block
    for source in rationale_sources:
        block = transfer_section.split(f'"{source}"', 1)[1].split("},", 1)[0]
        assert '"prompt_variant": "explanation_then_answer"' in block
        assert '"cot": True' in block


def test_paper_pipeline_final_manifest_requires_audit_artifacts():
    text = Path("notebooks/00_paper_pipeline.py").read_text()
    for name in [
        "experiment_ledger.csv",
        "split_leakage_audit.csv",
        "selected_split_records.csv",
        "uncertainty_estimates.csv",
        "claim_to_evidence_map.csv",
        "visual_qa_checklist.csv",
        "paper_pipeline_code_digest.json",
        "masking_examples.json",
        "rq2_retrieval_comparison.json",
    ]:
        assert name in text


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


def test_length_aware_alpha_matches_shifted_span_counts():
    from src.config.schema import ObjectiveConfig
    from src.data.constants import SPAN_ANSWER, SPAN_EXPLANATION, SPAN_IGNORE

    labels = torch.tensor([[-100, 10, 11, 12, 13, 14]])
    span_ids = torch.tensor(
        [[SPAN_IGNORE, SPAN_EXPLANATION, SPAN_EXPLANATION, SPAN_EXPLANATION, SPAN_ANSWER, SPAN_ANSWER]]
    )
    cfg = ObjectiveConfig(
        name="explanation_aware",
        alpha_mode="length_aware",
        answer_weight_multiplier=1.0,
    )
    n_expl, n_answer = supervised_span_counts(labels, span_ids)
    assert (n_expl, n_answer) == (3, 2)
    assert abs(effective_answer_alpha(cfg, labels, span_ids).item() - 0.4) < 1e-6


def test_length_aware_alpha_multiplier_upweights_answer_span():
    from src.config.schema import ObjectiveConfig
    from src.data.constants import SPAN_ANSWER, SPAN_EXPLANATION, SPAN_IGNORE

    labels = torch.tensor([[-100, 10, 11, 12, 13, 14]])
    span_ids = torch.tensor(
        [[SPAN_IGNORE, SPAN_EXPLANATION, SPAN_EXPLANATION, SPAN_EXPLANATION, SPAN_ANSWER, SPAN_ANSWER]]
    )
    cfg = ObjectiveConfig(
        name="explanation_aware",
        alpha_mode="length_aware",
        answer_weight_multiplier=2.0,
    )
    # weighted answer = 2 * 2 tokens; explanation = 3 tokens -> 4 / 7
    assert abs(effective_answer_alpha(cfg, labels, span_ids).item() - (4 / 7)) < 1e-6


def test_length_aware_alpha_missing_spans_is_finite():
    from src.config.schema import ObjectiveConfig
    from src.data.constants import SPAN_IGNORE

    cfg = ObjectiveConfig(name="explanation_aware", alpha_mode="length_aware")
    labels = torch.full((1, 4), -100)
    span_ids = torch.full((1, 4), SPAN_IGNORE)
    alpha = effective_answer_alpha(cfg, labels, span_ids)
    assert torch.isfinite(alpha)
    assert alpha.item() == 0.5


def test_length_aware_alpha_equals_uniform_token_ce_weighting():
    from src.config.schema import ObjectiveConfig
    from src.data.constants import SPAN_ANSWER, SPAN_EXPLANATION, SPAN_IGNORE

    torch.manual_seed(0)
    B, T, V = 1, 7, 11
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100
    span_ids = torch.tensor(
        [[SPAN_IGNORE, SPAN_EXPLANATION, SPAN_EXPLANATION, SPAN_EXPLANATION, SPAN_ANSWER, SPAN_ANSWER, SPAN_ANSWER]]
    )
    cfg = ObjectiveConfig(name="explanation_aware", alpha_mode="length_aware")
    alpha = effective_answer_alpha(cfg, labels, span_ids)
    l_answer = masked_token_ce(logits, labels, mask=span_ids == SPAN_ANSWER)
    l_expl = masked_token_ce(logits, labels, mask=span_ids == SPAN_EXPLANATION)
    natural = alpha * l_answer + (1 - alpha) * l_expl
    full = masked_token_ce(
        logits,
        labels,
        mask=(span_ids == SPAN_ANSWER) | (span_ids == SPAN_EXPLANATION),
    )
    assert torch.allclose(natural, full, atol=1e-6)


def test_invalid_alpha_mode_raises():
    from src.config.schema import ObjectiveConfig
    from src.data.constants import SPAN_ANSWER

    cfg = ObjectiveConfig(name="explanation_aware", alpha_mode="bad")
    labels = torch.tensor([[-100, 1]])
    span_ids = torch.tensor([[0, SPAN_ANSWER]])
    try:
        effective_answer_alpha(cfg, labels, span_ids)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid alpha_mode should raise ValueError")


def test_mask_region_replaces_only_requested_cell():
    img = Image.new("RGB", (4, 4), (255, 255, 255))
    masked = mask_region(img, 1, 0, rows=2, cols=2)
    assert img.getpixel((0, 2)) == (255, 255, 255)
    assert masked.getpixel((0, 2)) == (128, 128, 128)
    assert masked.getpixel((3, 0)) == (255, 255, 255)


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


def test_open_ended_answer_cleanup_stops_repeated_answer_loops():
    from src.evaluation.scoring import clean_generated_answer, split_reasoning_answer

    _, answer = split_reasoning_answer(
        "Answer: 2,500.00. Answer: Answer: Answer: Answer:"
    )
    assert answer == "2,500.00"

    _, answer = split_reasoning_answer(
        "Reasoning: I read the invoice. Answer: Regional Medical Program. Answer:"
    )
    assert answer == "Regional Medical Program"

    _, answer = split_reasoning_answer(
        "2,500.00. Answer: Answer: Answer: Answer:"
    )
    assert answer == "2,500.00"

    assert clean_generated_answer("KEEP IT") == "KEEP IT"


def test_per_row_neg_ce_matches_per_row_reference():
    # The anchor for batched eval: per_row_neg_ce over a B-row batch must equal
    # B independent single-item scores. HF's model(labels=…).loss for ONE row is
    # the mean CE over its non-masked shifted tokens — which is exactly what
    # cross_entropy(..., ignore_index=-100) returns per row. So if our batched
    # helper matches that per-row reference, batched and per-item eval agree.
    from torch.nn.functional import cross_entropy

    from src.evaluation.scoring import per_row_neg_ce

    torch.manual_seed(0)
    B, T, V = 3, 9, 13
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    for i, ctx in enumerate([2, 4, 5]):  # a different context-prefix length per row
        labels[i, :ctx] = -100

    got = per_row_neg_ce(logits, labels)
    for i in range(B):
        ref = cross_entropy(
            logits[i, :-1, :].float(), labels[i, 1:], ignore_index=-100
        ).item()
        assert abs(got[i] - (-ref)) < 1e-5, (got[i], -ref)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\n{len(tests)} core tests passed.")


if __name__ == "__main__":
    _run_all()
