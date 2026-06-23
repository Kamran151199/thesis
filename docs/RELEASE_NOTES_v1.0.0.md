# Release Notes: v1.0.0 Thesis/Journal Reproducibility Package

Suggested GitHub release tag:

```text
v1.0.0-thesis-journal
```

Suggested release title:

```text
Thesis and journal reproducibility package
```

Suggested release description:

```text
This release preserves the code, experiment configurations, Colab pipeline,
final figures, and LaTeX sources used for the thesis/paper:

Reasoning and Grounded Explanation Generation Using Vision-Language Models

The repository contains:

- source code for dataset loading, model wrappers, objectives, training,
  evaluation, masking diagnostics, and artifact generation;
- experiment YAML files for the completed BLIP-2 and Qwen2-VL runs;
- the Colab paper pipeline notebook;
- final generated figures used in the paper;
- clean thesis and journal LaTeX sources;
- reproducibility instructions and citation metadata.

The large runtime outputs are not included in git:

- model checkpoints;
- downloaded dataset caches;
- Google Drive paper artifact bundles;
- rendered PDF review files.

The experiments use public datasets cited in the manuscript. The study is a
controlled single-GPU protocol with capped subsets and one seed per completed
configuration, not a public leaderboard submission.
```

Recommended release checklist:

```text
1. Confirm git status is clean except intentional release files.
2. Run local CPU-safe tests.
3. Confirm clean-docs builds or Overleaf builds.
4. Create the tag v1.0.0-thesis-journal.
5. Create a GitHub release from that tag.
6. Optionally archive the GitHub release on Zenodo and use the DOI in the paper.
```
