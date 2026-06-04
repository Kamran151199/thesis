Verifier agent result: **APPROVED** after one revision. The main correction from the verifier was important: **length-aware alpha is not automatically a new explanation-aware method**; it is basically the natural token-weighted CE control unless we add an explicit span multiplier.

**Approved Completion Checklist**

**A. Research Logic**
1. **Rewrite the central claim**: controlled, resource-constrained VLM adaptation, not “explanation-aware is best.”  
Reason: answer-only is strong, explanation-aware is conditional, BLIP-2 contrastive is negative.

2. **Fix RQ1**: make it clearly literature-synthesis only, or add a short “RQ1 synthesis” paragraph after Literature Review.  
Reason: no experiment answers RQ1.

3. **Limit RQ3 to rationale-bearing datasets**: ScienceQA and A-OKVQA.  
Reason: ChartQA, DocVQA, VQAv2 have no gold rationales.

4. **Fix RQ4**: current Table 6/Figure 7 are in-domain fine-tuning, not factor analysis.  
Reason: they do not isolate architecture/training factors.

5. **Fix RQ5 wording**: say evidence sensitivity/masking drift, not hallucination reduction.  
Reason: no hallucination annotation protocol exists.

6. **Fix RQ6**: either narrow to “2B vs 7B under one fixed objective” or run all objectives at 7B.  
Reason: current scale evidence is only explanation-aware `alpha=0.5`.

7. **Clarify RQ7**: transfer matrix rows are separate adapters, not one model.  
Reason: each source row has different dataset/objective.

**B. Objective / Alpha**
8. **Add `alpha_mode: fixed | length_aware`**.  
Reason: current fixed `0.5` overweights short answer spans.

9. **Treat length-aware alpha as natural CE control, not automatically the main explanation-aware method.**  
Reason: `alpha = answer_tokens / total_tokens` is near-equivalent to uniform rationale-generative CE.

10. **Add explicit span multiplier if we want true explanation-aware weighting.**  
Example: `answer_weight_multiplier` or `span_weight_ratio`.  
Reason: separates natural length correction from deliberate answer/explanation reweighting.

11. **Keep fixed-alpha sweep as ablation.**  
Reason: useful to study span-weight sensitivity.

12. **Log answer token count, explanation token count, effective alpha, multiplier.**  
Reason: paper must justify alpha behavior.

13. **Add tests for alpha modes.**  
Reason: objective weighting is core method logic.

**C. Experiments**
14. **Add/identify natural CE runs for ScienceQA and A-OKVQA.**  
Reason: this is the principled control beside fixed-alpha runs.

15. **Keep answer-only and rationale-generative controls.**  
Reason: they show whether rationales help beyond answer-only training.

16. **Add a selected explanation-aware A-OKVQA run under the final alpha policy.**  
Reason: A-OKVQA currently only has fixed `0.5`.

17. **For RQ6, either add 7B answer-only/rationale/selected explanation-aware or narrow the claim.**  
Reason: current broad QLoRA wording is not supported.

18. **Add BLIP-2 retrieval diagnostics for frozen, generative, contrastive, and random baseline.**  
Reason: current contrastive-only retrieval plot is not interpretable enough.

19. **Do not claim explanation-aware training on no-rationale datasets.**  
Reason: no gold rationales exist there.

20. **Keep ChartQA/DocVQA/VQAv2 as answer-only domain/transfer experiments.**  
Reason: useful, but different claim boundary.

21. **Add experiment ledger before Results.**  
Include backbone, dataset, split/cap, rationale availability, objective, alpha mode, prompt family, metric, RQ.  
Reason: readers need design before results.

22. **Fix confusing run names/display aliases.**  
Reason: `rq4_qwen2vl_explanation_aware_docvqa` is actually answer-only fallback.

**D. Evaluation**
23. **Define “headline metric.”**  
Reason: it means each dataset’s native main metric, not a universal comparable score.

24. **Report native metrics beside headline.**  
Reason: ANLS, relaxed accuracy, MC accuracy, VQA accuracy mean different things.

25. **Do not report ROUGE/BLEU for runs without generated rationales.**  
Reason: BLIP-2/answer-only runs do not have comparable rationale outputs.

26. **Add random/zero-shot baselines.**  
Reason: makes gains interpretable.

27. **Add uncertainty estimates.**  
Use binomial SE or bootstrap CI.  
Reason: tiny differences like `0.790` vs `0.795` may be noise.

28. **Check evaluation fairness.**  
Either add shared evaluation mode or clearly state each model is evaluated with its trained prompt family.  
Reason: answer-only vs rationale prompts can confound comparison.

**E. Data Integrity**
29. **Audit all splits/caps/sampling.**  
Reason: capped subsets must be reproducible.

30. **Verify no train/eval leakage.**  
Especially DocVQA and VQAv2.  
Reason: strong DocVQA result needs trust.

31. **Save selected train/eval IDs.**  
Reason: reproducibility.

32. **Document rationale availability per dataset.**  
Reason: central to claim boundaries.

**F. Masking / Faithfulness**
33. **Show masking examples.**  
Original image, masked image, question, gold answer, full score, masked score, drift.  
Reason: readers need to see what masking means.

34. **Add one ChartQA or DocVQA masking figure.**  
Reason: structured evidence is visually inspectable.

35. **Add masking controls.**  
Random region, top-drift region, optionally blank/full-image.  
Reason: mean drift alone is weak.

36. **Verify masked image reaches the model after preprocessing.**  
Reason: prevents silent perturbation failure.

37. **Call it evidence sensitivity, not proof of faithful rationales.**  
Reason: drift measures answer likelihood change, not explanation grounding.

38. **Warn drift is not directly comparable across datasets.**  
Reason: answer length/task format affects likelihood scale.

39. **Explain low ScienceQA drift.**  
Reason: may reflect language priors, MC options, diffuse evidence, coarse masks.

40. **State masking scores gold-answer likelihood.**  
Reason: not generated-answer change.

**G. Figures / Tables**
41. **Add sample input construction figure/table.**  
Raw sample → prompt → target → model-facing text → supervised spans.  
Reason: answers your “what does model see?” concern.

42. **Keep full token/collator trace as artifact/appendix.**  
Reason: too detailed for main paper.

43. **Fix Figure 4 retrieval comparison.**  
Frozen/generative/contrastive/random.  
Reason: contrastive-only plot lacks reference.

44. **Fix Figure 5 objective comparison.**  
Include zero-shot, answer-only, rationale-generative/natural CE, explanation-aware.  
Reason: answer-only is the strongest control.

45. **Rename Table 6/Figure 7 as in-domain adaptation.**  
Reason: not cross-domain.

46. **Use transfer table/heatmap only for transfer claims.**  
Reason: transfer means train on one dataset, evaluate another.

47. **Add captions stating rationale vs answer-only fallback.**  
Reason: prevents overclaiming.

48. **Fix float placement.**  
Use `FloatBarrier` or move floats near text.  
Reason: current figures/tables drift too far.

49. **Visual QA all figures.**  
No overlapping text, cropped axes, unreadable grids, pixelation.

**H. Paper Text**
50. **Replace “this thesis” with “this study/paper.”**  
Reason: journal manuscript tone.

51. **Remove “2024–2025” wording.**  
Reason: artificial.

52. **Rewrite abstract.**  
Must include BLIP-2 generative, BLIP-2 contrastive, Qwen answer-only, rationale-generative, explanation-aware, scale, transfer, masking.  
Reason: current abstract omits key controls.

53. **Rewrite highlights.**  
Each highlight should be one clear result.  
Reason: current ones mix unrelated claims.

54. **Rewrite Results opening.**  
Reason: current artifact-validation paragraph is weird.

55. **Remove “Colab logs.”**  
Reason: implementation detail.

56. **Fix declarations.**  
Authors wrote/reviewed paper; AI only assisted language/code; add funding/conflict/data/code/artifact availability.  
Reason: journal compliance.

57. **Keep claims bachelor-level.**  
Reason: capped data/single seed cannot support broad SOTA claims.

**I. References**
58. **Add dataset citations.**  
DocVQA, ChartQA, ScienceQA, A-OKVQA, VQAv2.

59. **Add method citations.**  
BLIP-2, Qwen2-VL, LoRA, QLoRA. Use DoRA only as related work if not implemented.

60. **Add metric citations where needed.**  
BLEU, ROUGE, ANLS/VQA scoring, or benchmark metric definitions.

61. **Remove unused refs.**  
Especially Qwen2.5-VL/PaliGemma if not used.

62. **Verify every BibTeX entry from official source.**  
Reason: current bib still has arXiv/venue-risk entries.

63. **Add citations in Methodology, not only Literature Review.**  
Reason: methods/metrics/backbones need support where introduced.

**J. Pipeline / Code Robustness**
64. **Expose alpha mode and span multiplier in config/notebook/artifacts.**  
Reason: reproducibility.

65. **Add artifact manifest entries for experiment ledger, masking examples, uncertainty tables, claim-evidence map.**  
Reason: auditability.

66. **Invalidate stale results when alpha/config/code changes.**  
Reason: avoid reusing old runs.

67. **Add prompt/template tests.**  
Reason: answer-only must not contain rationale.

68. **Add transfer prompt-mode tests.**  
Reason: answer-only and rationale sources need correct eval prompts.

69. **Add BLIP-2 contrastive preflight.**  
Reason: fragile branch.

70. **Save figures as PDF and PNG; include PDF in LaTeX.**  
Reason: no pixelation.

71. **Keep large zips out of git or document separately.**  
Reason: repo hygiene.

72. **Fix qualitative generation/decoding or exclude grids.**  
Reason: unreadable qualitative figures damage credibility.

**K. Final Bachelor-Level Boundary**
73. **Do not require full SOTA-scale/full-dataset/multi-seed reruns.**  
Reason: bachelor scope.

74. **Minimum reruns after logic fixes:** ScienceQA/A-OKVQA alpha/natural-control refresh, BLIP retrieval baselines, masking examples, refreshed figures/tables.  
Reason: fixes core logic gaps.

75. **Optional stronger reruns:** 7B all-objective comparison; teacher rationales for no-rationale datasets.  
Reason: useful but heavier.

76. **Final technical gate:** notebook top-to-bottom, no missing/stale/failed runs, PDF builds, no undefined refs/cites/overfull boxes.

77. **Final visual gate:** inspect every included figure/table manually.

78. **Final claim-to-evidence map:** every abstract/highlight/conclusion claim must point to a table/figure/result row.

This is the approved list I would use as the actual completion roadmap.