# Thesis Notes

## Structure

```
notes/
├── papers/          # One .md per paper you read (copy _TEMPLATE.md)
├── videos/
│   ├── transcripts/ # Auto-downloaded YouTube transcripts (timestamped)
│   └── *.md         # Your notes per video (copy _TEMPLATE.md)
├── concepts/        # Quick reference for concepts you learn (copy _TEMPLATE.md)
└── README.md
```

## Workflow for Videos

### 1. Download the transcript

```bash
cd notes/videos
./download-transcript.sh "https://www.youtube.com/watch?v=kCc8FmEb1nY" "karpathy-gpt-from-scratch"
```

This saves a timestamped transcript to `transcripts/karpathy-gpt-from-scratch.txt`.

### 2. Watch the video, take sparse notes

Copy `_TEMPLATE.md` to a new file (e.g., `karpathy-gpt-from-scratch.md`).
Jot down timestamps and brief notes as you watch.

### 3. Ask Claude about any point

Just tell Claude something like:

> "Read the transcript at notes/videos/transcripts/karpathy-gpt-from-scratch.txt.
> At around 45:00 he talks about residual connections. Explain that deeper
> and how it relates to VLMs."

Claude will read the transcript, find the context around that timestamp,
and give you a deep explanation connected to your thesis.

## Workflow for Papers

1. Copy `papers/_TEMPLATE.md` to `papers/vaswani-2017-attention.md` (or similar)
2. Fill in as you read — keep it brief, max 10-15 bullet points
3. Focus on "What I Learned" and "How This Connects to My Thesis"

## Workflow for Concepts

Use `concepts/` for things you want to reference quickly later:
- `concepts/self-attention.md`
- `concepts/cross-attention.md`
- `concepts/qlora.md`
- etc.

Keep these short — 5-10 lines max. The thesis is your real knowledge artifact.
