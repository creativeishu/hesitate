"""
Hesitate — visualise token-by-token generation of a language model.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hesitate",
    page_icon="🔍",
    layout="wide",
)

st.title("Hesitate")
st.caption("Watch a language model choose the next token, one step at a time.")

# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

MODELS = {
    "GPT-2 Medium (345M)":               "gpt2-medium",
    "GPT-2 Large (774M)":                "gpt2-large",
    "GPT-2 (small, 117M)":               "gpt2",
    "DistilGPT-2 (82M)":                 "distilgpt2",
    "Phi-2 (2.7B) — recommended":        "microsoft/phi-2",
    "Phi-3-mini (3.8B)":                 "microsoft/Phi-3-mini-4k-instruct",
    "Llama-3.2-3B (needs HF token)":     "meta-llama/Llama-3.2-3B",
    "Custom (type below)":               "__custom__",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32


def load_model(model_id: str):
    with st.status("Loading model…", expanded=True) as status:
        st.write(f"Device: **{DEVICE}** | dtype: **{DTYPE}**")
        st.write("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        st.write("Loading model weights…")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=DTYPE, device_map=DEVICE
        )
        st.write("Finalising…")
        model.eval()
        status.update(label="Model ready!", state="complete", expanded=False)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Zero out tokens outside the nucleus (top-p).
    Sorts descending, computes cumulative sum, masks tokens beyond the cutoff,
    then renormalises. The top token is always kept regardless of top_p.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Remove tokens where cumulative prob *exceeds* top_p, but keep the first
    # token that pushes cumulative over top_p (so we always have at least one)
    remove = (cumulative - sorted_probs) >= top_p
    sorted_probs[remove] = 0.0

    # Scatter back to original order and renormalise
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx] = sorted_probs
    filtered = filtered / filtered.sum()
    return filtered


def get_next_token_distribution(
    tokenizer, model, text: str, temperature: float = 1.0, top_p: float = 1.0
):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    if temperature <= 0.01:
        # Fully greedy
        probs_all = torch.zeros_like(logits)
        probs_all[logits.argmax()] = 1.0
        selected_id = int(logits.argmax())
    else:
        probs_all = torch.softmax(logits / temperature, dim=-1)

        # Apply top-p nucleus truncation before sampling
        if top_p < 1.0:
            probs_all = _apply_top_p(probs_all, top_p)

        selected_id = int(torch.multinomial(probs_all, num_samples=1))

    # Always show top-20 of the (possibly nucleus-truncated) distribution
    top = torch.topk(probs_all, k=min(20, int((probs_all > 0).sum())))
    top_tokens = [tokenizer.decode([tid]) for tid in top.indices.tolist()]
    top_probs  = top.values.tolist()

    selected_token = tokenizer.decode([selected_id])
    selected_prob  = float(probs_all[selected_id])

    # Number of tokens in the nucleus
    n_nucleus = int((probs_all > 0).sum())

    entropy_bits = float(-torch.sum(probs_all * torch.log2(probs_all + 1e-12)))

    return top_tokens, top_probs, selected_token, selected_prob, entropy_bits, n_nucleus


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_histogram(tokens, probs, selected_token) -> go.Figure:
    colours = ["#e63946" if t == selected_token else "#457b9d" for t in tokens]
    tokens_r  = tokens[::-1]
    probs_r   = probs[::-1]
    colours_r = colours[::-1]

    fig = go.Figure(go.Bar(
        x=probs_r,
        y=[repr(t) for t in tokens_r],
        orientation="h",
        marker_color=colours_r,
        text=[f"{p*100:.1f}%" for p in probs_r],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Token",
        xaxis=dict(range=[0, max(probs) * 1.25]),
        margin=dict(l=10, r=80, t=10, b=40),
        height=520,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white", size=13),
        xaxis_color="white",
        yaxis_color="white",
    )
    return fig


def render_sentence(prompt: str, history: list[tuple[str, float]]) -> str:
    parts = [f'<span style="color:white;">{prompt}</span>']
    for token, prob in history:
        colour = "#2dc653" if prob > 0.5 else "#f4a261" if prob > 0.2 else "#e63946"
        parts.append(f'<span style="color:{colour}; font-weight:bold;">{token}</span>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

defaults = dict(
    sentence="",
    history=[],
    started=False,
    top_tokens=[],
    top_probs=[],
    selected_token="",
    selected_prob=0.0,
    entropy=0.0,
    n_nucleus=0,
    model_ready=False,
    model_label="",
    tokenizer=None,
    model=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model")

    chosen_label = st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
        index=4,  # Phi-2 default
        disabled=st.session_state.model_ready,
    )
    model_id = MODELS[chosen_label]

    if model_id == "__custom__":
        model_id = st.text_input(
            "HuggingFace model ID",
            placeholder="e.g. microsoft/phi-2",
            disabled=st.session_state.model_ready,
        ).strip()

    if not st.session_state.model_ready:
        if st.button(
            "Load model",
            use_container_width=True,
            type="primary",
            disabled=not model_id or model_id == "__custom__",
        ):
            tokenizer, model = load_model(model_id)
            st.session_state.tokenizer  = tokenizer
            st.session_state.model      = model
            st.session_state.model_ready = True
            st.session_state.model_label = chosen_label

    if st.session_state.model_ready:
        st.success(f"✓ {st.session_state.model_label}")
        if st.button("Load different model", use_container_width=True):
            for key in ("model_ready", "started"):
                st.session_state[key] = False
            for key in ("tokenizer", "model"):
                st.session_state[key] = None
            st.session_state.history = []
            st.rerun()

    st.caption(
        "Smaller models (distilgpt2, gpt2) lack factual knowledge. "
        "GPT-2 Medium is the minimum for sensible factual completions."
    )

    st.divider()
    st.header("Sampling settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.1, max_value=2.0, value=1.0, step=0.05,
        help=(
            "Divides logits before softmax.\n\n"
            "< 1.0 → distribution peaks sharply at top token\n\n"
            "= 1.0 → raw model output\n\n"
            "> 1.0 → distribution flattens, more surprising picks"
        ),
    )
    st.caption(
        f"{'🥶 Deterministic' if temperature <= 0.2 else '❄️ Focused' if temperature < 0.8 else '🌡️ Balanced' if temperature <= 1.2 else '🔥 Creative' if temperature <= 1.7 else '🌋 Chaotic'}"
        f"  —  T = {temperature:.2f}"
    )

    top_p = st.slider(
        "Top-p (nucleus sampling)",
        min_value=0.1, max_value=1.0, value=1.0, step=0.05,
        help=(
            "Keeps only the smallest set of tokens whose cumulative "
            "probability exceeds p, then renormalises and samples from those.\n\n"
            "1.0 → sample from the full vocabulary (no truncation)\n\n"
            "0.9 → discard the low-probability tail, keep 90% of the mass\n\n"
            "0.5 → nucleus is very tight, only the most likely tokens survive"
        ),
    )
    if top_p < 1.0:
        st.caption(f"Nucleus: top tokens covering {top_p*100:.0f}% of probability mass")
        if st.session_state.n_nucleus:
            st.caption(f"→ {st.session_state.n_nucleus} tokens in nucleus at last step")
    else:
        st.caption("Full vocabulary — no nucleus truncation")

    st.divider()
    st.markdown(
        "**Token colour guide**\n\n"
        "🟢 > 50% confident\n\n"
        "🟠 20–50%\n\n"
        "🔴 < 20% (hesitating)"
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

def _run_inference():
    return get_next_token_distribution(
        st.session_state.tokenizer,
        st.session_state.model,
        st.session_state.sentence,
        temperature,
        top_p,
    )

with col_left:
    st.subheader("Input")

    prompt = st.text_area(
        "Enter a prompt",
        placeholder="The capital of France is",
        height=80,
        disabled=not st.session_state.model_ready,
    )

    btn_start, btn_next, btn_reset = st.columns(3)

    with btn_start:
        if st.button(
            "▶ Start",
            use_container_width=True,
            type="primary",
            disabled=not st.session_state.model_ready,
        ):
            if prompt.strip():
                st.session_state.sentence = prompt.strip()
                st.session_state.history  = []
                st.session_state.started  = True
                (
                    st.session_state.top_tokens,
                    st.session_state.top_probs,
                    st.session_state.selected_token,
                    st.session_state.selected_prob,
                    st.session_state.entropy,
                    st.session_state.n_nucleus,
                ) = _run_inference()

    with btn_next:
        if st.button(
            "⏭ Next",
            use_container_width=True,
            disabled=not (st.session_state.model_ready and st.session_state.started),
        ):
            st.session_state.sentence += st.session_state.selected_token
            st.session_state.history.append(
                (st.session_state.selected_token, st.session_state.selected_prob)
            )
            (
                st.session_state.top_tokens,
                st.session_state.top_probs,
                st.session_state.selected_token,
                st.session_state.selected_prob,
                st.session_state.entropy,
                st.session_state.n_nucleus,
            ) = _run_inference()

    with btn_reset:
        if st.button("↺ Reset", use_container_width=True):
            for key in ("sentence", "selected_token", "model_label"):
                pass
            st.session_state.sentence       = ""
            st.session_state.history        = []
            st.session_state.started        = False
            st.session_state.top_tokens     = []
            st.session_state.top_probs      = []
            st.session_state.selected_token = ""
            st.session_state.selected_prob  = 0.0
            st.session_state.entropy        = 0.0
            st.session_state.n_nucleus      = 0

    st.divider()
    st.subheader("Generated sentence")

    if st.session_state.sentence:
        st.markdown(
            f'<p style="font-size:18px; line-height:1.8;">'
            f'{render_sentence(prompt.strip(), st.session_state.history)}'
            f'</p>',
            unsafe_allow_html=True,
        )

    if st.session_state.started:
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Selected token",
            repr(st.session_state.selected_token),
            f"{st.session_state.selected_prob*100:.1f}%",
        )
        m2.metric(
            "Entropy",
            f"{st.session_state.entropy:.2f} bits",
            help="Higher = more uncertain.",
        )
        m3.metric(
            "Nucleus size",
            st.session_state.n_nucleus,
            help="Number of tokens in the nucleus after top-p truncation.",
        )

with col_right:
    st.subheader("Next token distribution")
    if top_p < 1.0:
        st.caption(f"Showing top 20 of the {st.session_state.n_nucleus or '?'}-token nucleus (p={top_p})")
    else:
        st.caption("Showing top 20 tokens from full vocabulary")

    if st.session_state.top_tokens:
        fig = make_histogram(
            st.session_state.top_tokens,
            st.session_state.top_probs,
            st.session_state.selected_token,
        )
        st.plotly_chart(fig, use_container_width=True)
    elif not st.session_state.model_ready:
        st.info("Load a model from the sidebar to get started.")
    else:
        st.info("Enter a prompt and click ▶ Start to begin.")
