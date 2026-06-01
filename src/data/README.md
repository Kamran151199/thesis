# `src/data/` — datasets, prompts, and the masking collator

## Mental model

Five datasets with incompatible columns all funnel into **one** neutral type,
`VLMExample`, so nothing downstream branches on dataset name:

```
HF rows ──to_example()──► VLMExample(image, question, answer, choices?, answer_index?, explanation?)
                              │
              PromptTemplate  │  (prompt, target)        ← the RQ3 lever
                              ▼
          VLMCollator ──► batch: input_ids, pixel_values, labels (prompt masked), span_ids?
```

## The three pieces

| File | Role |
|------|------|
| `example.py` | `VLMExample` — the universal sample. `is_multiple_choice` / `has_explanation` are the only branches downstream needs. |
| `prompts.py` | prompt templates. `answer_only` vs `explanation_then_answer` **is** the RQ3 experiment. `target_spans` splits the target so the loss can weight answer vs reasoning. |
| `collator.py` | turns examples into a padded batch and **masks the prompt to −100** so loss is target-only. The most bug-prone file — read its top docstring. |
| `base.py` | `BaseVLMDataset` — shared load/filter/subsample; subclasses supply `_load_raw` + `to_example`. |
| `datasets/` | one loader per benchmark. ScienceQA is proven; others document their HF schema. |

## The masking trick (why the collator asks the *wrapper*, not the tokenizer)

`prompt_len` must include the image placeholder tokens the processor inserts (32
for BLIP-2; variable for Qwen2-VL). Counting text tokens alone undercounts → the
prompt tail leaks into the supervised target (a real bug we hit). So the collator
calls `wrapper.input_length(image, prompt)`, which encodes through the backbone's
processor *with the image*. Backbone-specific image-token logic stays on the
wrapper; the collator stays generic.

## Add a dataset

1. New file in `datasets/`, subclass `BaseVLMDataset`, set `hf_path` /
   `is_multiple_choice` / `has_gold_explanation`.
2. Implement `_load_raw(split)` and `to_example(row)` (and `_keep` to filter).
3. `@DATASETS.register("yourname")` on the class.
4. Add it to the import block in `datasets/__init__.py`.

## Add a prompt variant (new RQ3 arm)

Subclass `PromptTemplate`, override `target` (and `target_spans` if it has
distinct answer/explanation spans), `@PROMPT_VARIANTS.register("name")`.
