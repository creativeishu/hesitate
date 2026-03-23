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
    "GPT-2 Medium (345M) ★ recommended": "gpt2-medium",
    "GPT-2 Large (774M)":                "gpt2-large",
    "GPT-2 (small, 117M)":               "gpt2",
    "DistilGPT-2 (82M, fastest)":        "distilgpt2",
}

# ---------------------------------------------------------------------------
# Model loading — synchronous, inside st.status() for live feedback
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """Load tokenizer and model, showing live status in the sidebar."""
    with st.status("Loading model…", expanded=True) as status:
        st.write("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        st.write("Loading model weights…")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

        st.write("Finalising…")
        model.eval()

        status.update(label="Model ready!", state="complete", expanded=False)

    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_next_token_distribution(tokenizer, model, text: str, top_k: int = 20):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    probs_all = torch.softmax(logits, dim=-1)

    top = torch.topk(probs_all, k=top_k)
    top_tokens = [tokenizer.decode([tid]) for tid in top.indices.tolist()]
    top_probs = top.values.tolist()

    selected_id = int(torch.argmax(probs_all))
    selected_token = tokenizer.decode([selected_id])
    selected_prob = float(probs_all[selected_id])

    entropy_bits = float(-torch.sum(probs_all * torch.log2(probs_all + 1e-12)))

    return top_tokens, top_probs, selected_token, selected_prob, entropy_bits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_histogram(tokens, probs, selected_token) -> go.Figure:
    colours = ["#e63946" if t == selected_token else "#457b9d" for t in tokens]
    tokens_r, probs_r, colours_r = tokens[::-1], probs[::-1], colours[::-1]

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
        index=0,
        disabled=st.session_state.model_ready,
    )
    model_id = MODELS[chosen_label]

    if not st.session_state.model_ready:
        if st.button("Load model", use_container_width=True, type="primary"):
            tokenizer, model = load_model(model_id)
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_ready = True
            st.session_state.model_label = chosen_label

    if st.session_state.model_ready:
        st.success(f"✓ {st.session_state.model_label}")
        if st.button("Load different model", use_container_width=True):
            st.session_state.model_ready = False
            st.session_state.tokenizer = None
            st.session_state.model = None
            st.session_state.started = False
            st.session_state.history = []
            st.rerun()

    st.caption(
        "Smaller models (distilgpt2, gpt2) lack factual knowledge — "
        "they won't reliably predict 'Paris' for 'capital of France'. "
        "GPT-2 Medium is the minimum size for sensible factual completions."
    )
    st.divider()
    st.header("Settings")
    top_k = st.slider("Top-k tokens to display", min_value=5, max_value=40, value=20)
    st.divider()
    st.markdown(
        "**Colour guide (generated tokens)**\n\n"
        "🟢 > 50% confident\n\n"
        "🟠 20–50%\n\n"
        "🔴 < 20% (hesitating)"
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

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
                st.session_state.history = []
                st.session_state.started = True
                (
                    st.session_state.top_tokens,
                    st.session_state.top_probs,
                    st.session_state.selected_token,
                    st.session_state.selected_prob,
                    st.session_state.entropy,
                ) = get_next_token_distribution(
                    st.session_state.tokenizer,
                    st.session_state.model,
                    st.session_state.sentence,
                    top_k,
                )

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
            ) = get_next_token_distribution(
                st.session_state.tokenizer,
                st.session_state.model,
                st.session_state.sentence,
                top_k,
            )

    with btn_reset:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.sentence = ""
            st.session_state.history = []
            st.session_state.started = False
            st.session_state.top_tokens = []
            st.session_state.top_probs = []
            st.session_state.selected_token = ""
            st.session_state.selected_prob = 0.0
            st.session_state.entropy = 0.0

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
        m1, m2 = st.columns(2)
        m1.metric(
            "Selected token",
            repr(st.session_state.selected_token),
            f"{st.session_state.selected_prob*100:.1f}%",
        )
        m2.metric(
            "Entropy",
            f"{st.session_state.entropy:.2f} bits",
            help="Higher = more uncertain. Lower = more confident.",
        )

with col_right:
    st.subheader("Next token distribution")

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
