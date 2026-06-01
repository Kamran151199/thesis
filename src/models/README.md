# `src/models/` — backbone wrappers + QLoRA

## Mental model

Each VLM (BLIP-2, Qwen2-VL, PaliGemma) has a different class, processor, module
names and freeze policy. A **wrapper** hides all of that behind one interface so
the trainer/evaluator are backbone-blind:

```
wrapper.model       # the (4-bit + LoRA) HF module the optimizer trains
wrapper.processor   # image preprocessing + tokenizer
wrapper.build_inputs(images, texts)   # backbone-aware encoding (image tokens!)
wrapper.input_length(image, text)     # length incl. image tokens (for masking)
wrapper.forward(batch) / .generate(batch, …)
```

## The shared load pipeline (in `base.py`, inherited by every backbone)

```
ModelConfig
   │  build_bnb_config()      ← 4-bit NF4         (quantization.py)   the "Q"
   ▼
from_pretrained(quantization_config=…, torch_dtype=bf16, device_map=auto)
   │  _apply_freeze()         ← hard-freeze vision tower
   │  apply_qlora()           ← rank-16 adapters  (peft_lora.py)      the "LoRA"
   ▼
ready-to-train model
```

A backbone subclass supplies only four facts: `model_cls`, `processor_cls`,
`default_lora_targets`, `default_freeze`.

| File | Role |
|------|------|
| `base.py` | `BaseVLMWrapper` — the interface + shared load pipeline + encoding hooks |
| `quantization.py` | 4-bit NF4 `BitsAndBytesConfig` builder (the "Q") |
| `peft_lora.py` | rank-16 LoRA application via PEFT (the "LoRA") |
| `backbones/blip2.py` | **proven** backbone; trains Q-Former (or QLoRA on OPT) |
| `backbones/qwen2_vl.py` | primary backbone; prepends Qwen's vision-token block |
| `backbones/paligemma.py` | third backbone (SigLIP + Gemma) |

## Two adaptation modes

- **`use_qlora: true`** — 4-bit base + LoRA adapters (the thesis-spec PEFT path).
- **`use_qlora: false`** — no quantization/LoRA; whatever isn't in `freeze`
  trains at full precision (the proven BLIP-2 "train the Q-Former" mode).

## Add a backbone

1. New file in `backbones/`, subclass `BaseVLMWrapper`, set the four class attrs.
2. If its processor needs a chat template / special image-token handling,
   override `prompt_to_model_text` (see Qwen2-VL) — keep `prompt+target` a true
   prefix so the collator masking stays valid.
3. `@BACKBONES.register("name")`, add to `backbones/__init__.py`.
4. **First run:** decode one label batch and confirm only the target is
   supervised (the `# VERIFY` note in each wrapper).
