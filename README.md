# Hesitate

**Watch a language model think — one token at a time.**

Hesitate is an interactive visualisation tool that slows down text generation to human speed. Instead of seeing the final output instantly, you step through it token by token and see the full probability distribution the model considered before each choice.

![Hesitate screenshot placeholder](docs/screenshot.png)

---

## What it shows

Every time a language model generates the next word, it does not simply "know" the answer. It computes a probability distribution over its entire vocabulary — sometimes tens of thousands of tokens — and then samples from it. The result depends on temperature, nucleus size, and randomness.

Hesitate makes that process visible:

- **The sentence** grows one token at a time, each new token colour-coded by how confident the model was
- **The histogram** shows the top 15 candidates and their probabilities before each pick
- **The selected token** is highlighted in orange — it may not always be the most probable one
- **Entropy** tells you how uncertain the model was at each step (high entropy = spread-out distribution, the model was genuinely hesitating)
- **Nucleus size** tells you how many tokens survived the top-p cutoff

---

## Why this is interesting

A few things you will notice immediately:

**Confidence varies wildly across a sentence.** For "The capital of France is", the model is very confident about "Paris" (low entropy, steep distribution). For a connective word like "and" or "the", the distribution is flat — many tokens are equally plausible. You can see this directly in the histogram shape.

**Temperature reshapes the distribution, not just the output.** At T=0.5 the histogram spikes dramatically toward the top token. At T=1.8 it flattens — rank 5 or rank 10 tokens start getting picked. Watching this live is more intuitive than any textbook explanation.

**Top-p cuts the tail.** At p=0.9, only the tokens covering 90% of the probability mass survive. When the model is confident the nucleus might be 5 tokens. When it is uncertain it might be 300. The nucleus size metric shows this changing step by step.

**Model size matters for factual knowledge.** Small models (distilgpt2, gpt2) do not reliably predict "Paris" for "The capital of France is" — the factual knowledge is not baked in. GPT-Neo 1.3B and above get it right. The difference is immediately visible in the histogram.

---

## Installation

### Requirements

- Python 3.10+
- ~6–15 GB disk space for model weights (depending on which model you load)
- 8 GB RAM minimum; 16 GB recommended for models above 1B parameters

### Install

```bash
git clone https://github.com/creativeishu/hesitate.git
cd hesitate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model weights

By default, HuggingFace downloads model weights to `~/.cache/huggingface/hub/`. If you want to store them elsewhere (e.g. an external drive), set the `HF_HOME` environment variable before running:

```bash
export HF_HOME=/Volumes/MyDrive/huggingface
streamlit run app.py
```

Add this line to your `~/.zshrc` or `~/.bashrc` to make it permanent.

---

## How to use it

### 1. Load a model

Select a model from the sidebar dropdown and click **Load model**. The model is downloaded from HuggingFace on first use and cached locally for subsequent runs.

**Recommended models for this tool** (pure text completion — not instruction-tuned):

| Model | Size | Notes |
|-------|------|-------|
| `GPT-Neo 1.3B` | ~5 GB | Good balance of quality and speed |
| `GPT-Neo 2.7B` | ~10 GB | Noticeably smarter, needs 16 GB RAM |
| `GPT-2 Medium` | ~1.5 GB | Fastest to load, decent quality |
| `GPT-2 Large` | ~3 GB | Better factual recall than medium |

> **Note on instruct models** (Phi-2, Phi-3, Llama-instruct): these models were fine-tuned on question-answer datasets. They tend to generate structured Q&A patterns rather than natural text continuations, which makes them less interesting for this tool. Use them if you want to observe that behaviour specifically.

### 2. Enter a prompt

Type a sentence in the prompt box. Try something where the next word is interesting:

- `The capital of France is` — watch the model predict "Paris" with high confidence
- `Two plus two equals` — factual, should be confident
- `The weather today is` — ambiguous, distribution will be flat
- `Once upon a time there was a` — creative, many plausible continuations

Click **▶ Start**.

### 3. Step through generation

Click **⏭ Next token** to advance one step at a time.

- The sentence grows by one token
- The histogram updates to show the distribution for the next position
- The token colour in the sentence reflects confidence:
  - 🟢 Green — model was >50% confident
  - 🟡 Amber — model was 20–50% confident
  - 🔴 Red — model was <20% confident (genuinely hesitating)

Click **↺** to reset and try a new prompt.

### 4. Adjust sampling parameters

The sliders are live — you can change them between steps at any time:

**Temperature** controls how peaked or flat the distribution is:
- `0.1` — nearly deterministic, always picks the top token
- `1.0` — raw model output, default
- `2.0` — very flat, low-probability tokens get picked regularly

**Top-p (nucleus sampling)** truncates the tail of the distribution:
- `1.0` — sample from the full vocabulary (no truncation)
- `0.9` — keep only tokens covering 90% of probability mass
- `0.5` — very tight nucleus, only the most likely tokens survive

Try this experiment: start at T=1.0, p=1.0, step a few tokens. Then drag temperature to 1.8 — watch the histogram flatten dramatically. Then set p=0.5 — watch the nucleus shrink and the chart simplify.

---

## Metrics explained

| Metric | What it means |
|--------|--------------|
| **Selected token** | The token the model picked at this step (sampled from the distribution) |
| **Probability** | The probability assigned to the selected token under the current distribution |
| **Entropy** | Bits of uncertainty in the distribution. Low = confident (one token dominates). High = uncertain (many tokens are plausible) |
| **Nucleus size** | Number of tokens with non-zero probability after top-p truncation. At p=1.0 this is the full vocabulary size |

---

## Supported hardware

Hesitate auto-detects the best available device:

| Hardware | Detected as | Precision |
|----------|-------------|-----------|
| Apple Silicon (M1/M2/M3/M4) | MPS | float16 |
| NVIDIA GPU | CUDA | float16 |
| CPU (any) | CPU | float32 |

On Apple Silicon, models run on the unified GPU via Metal (MPS), which is significantly faster than CPU for inference.

---

## Stack

- [Streamlit](https://streamlit.io) — UI framework
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — model loading and inference
- [PyTorch](https://pytorch.org) — tensor operations and sampling
- [Plotly](https://plotly.com/python/) — interactive bar chart

---

## Known limitations

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) for a full list. Key points:

- **Model loading cannot be cancelled mid-download** — once a large model starts downloading, it must complete. Choose your model carefully before clicking Load.
- **Instruct models behave unexpectedly** — models fine-tuned for chat (Phi, Llama-instruct) generate Q&A structure rather than natural text. Use base/completion models.
- **No GPU support for very large models** — models above ~7B parameters will not fit in 16 GB of unified memory. Stick to the 1–3B range on a 16 GB machine.

---

## Roadmap

- [ ] Counterfactual branching — click any bar to pick that token instead and diverge
- [ ] Repetition penalty slider
- [ ] Token history panel — see all past distributions in a scrollable timeline
- [ ] Export session as a shareable report
- [ ] FastAPI + React frontend for better performance and true cancel support
