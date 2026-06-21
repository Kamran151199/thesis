# AGENTS.md

## Project

Bachelor's thesis: "Cross-Modal Generative Alignment for Document, Chart, and Image Reasoning with Explanation-Aware Vision-Language Models"

Timeline: 50 days starting 2026-04-16. Roadmap is in `final-roadmap.md`.

## Who I Am

Senior Software/Data Engineer (~10 years). Transitioning to ML/AI engineering. Have basic NLP experience. Learning transformers, ViT, VLMs, and alignment methods from the ground up. Fast learner.

## How to Explain Things to Me

I need "bent fingers" level understanding. That means:

- **Make it visual.** If a concept can be drawn, draw it (ASCII diagrams, matplotlib plots, analogies to physical things). Don't describe what a matrix does — show me the grid warping.
- **Make it concrete.** Don't say "the model learns representations." Say "after training, the vector for 'dog' and the vector for a photo of a dog end up pointing in roughly the same direction in 768-dimensional space — like two arrows agreeing."
- **Make it buildable.** Show me the 10-line Python snippet that makes the concept real. I understand code faster than math notation.
- **Use examples first, theory second.** Show me what happens, then explain why. Not the other way around.
- **No hand-waving.** If I ask "how does attention work?", don't say "it lets tokens attend to each other." Walk me through Q, K, V, the dot product, the softmax, the weighted sum — with actual numbers if needed.

If I can't close my eyes and picture the concept doing its thing, the explanation isn't done yet.

## Project Structure

```
exploration/     # Interactive .py files with # %% cells (learning)
src/             # Thesis implementation code (Weeks 4-6)
  data/          # Dataset loaders
  models/        # VLM wrappers, alignment objectives
  training/      # Fine-tuning scripts
  evaluation/    # Metrics and analysis
configs/         # Training configs (YAML)
notes/           # Paper notes, video transcripts, concept cards
thesis_doc/      # Written thesis (LaTeX/Word)
```

## Code Conventions

- Python files with `# %%` cell markers for interactive use in VS Code
- Use uv for package management (`uv run`, `uv add`)
- Format with Ruff
- Track experiments with wandb
- Keep exploration/ files self-contained and runnable

## Video Transcripts

Karpathy "Zero to Hero" transcripts are in `notes/videos/transcripts/`. When I reference a timestamp from a video, read the transcript file to find context and give deep explanations tied to my thesis.
