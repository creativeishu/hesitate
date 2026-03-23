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
# Model loading — cached so it only loads once
# ---------------------------------------------------------------------------

MODELS = {
    "GPT-2 (small, 117M)": "gpt2",
    "GPT-2 Medium (345M)": "gpt2-medium",
    "GPT-2 Large (774M)": "gpt2-large",
    "DistilGPT-2 (82M, fastest)": "distilgpt2",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference — get top-k token probabilities for the next token
# ---------------------------------------------------------------------------

def get_next_token_distribution(
    tokenizer,
    model,
    text: str,
    top_k: int = 20,
) -> tuple[list[str], list[float], str, float]:
    """
    Run a forward pass and return the top-k tokens and their probabilities.

    Returns
    -------
    tokens : list[str]
        Top-k token strings.
    probs : list[float]
        Corresponding probabilities (sum <= 1.0).
    selected_token : str
        The token the model would greedily select (argmax).
    selected_prob : float
        Probability of the selected token.
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]           # logits for next token
    probs_all = torch.softmax(logits, dim=-1)   # full vocabulary distribution

    top = torch.topk(probs_all, k=top_k)
    top_probs = top.values.tolist()
    top_ids = top.indices.tolist()
    top_tokens = [tokenizer.decode([tid]) for tid in top_ids]

    # Greedy selection
    selected_id = int(torch.argmax(probs_all))
    selected_token = tokenizer.decode([selected_id])
    selected_prob = float(probs_all[selected_id])

    # Entropy (nats → bits)
    entropy_bits = float(
        -torch.sum(probs_all * torch.log2(probs_all + 1e-12))
    )

    return top_tokens, top_probs, selected_token, selected_prob, entropy_bits


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_histogram(
    tokens: list[str],
    probs: list[float],
    selected_token: str,
) -> go.Figure:
    colours = [
        "#e63946" if t == selected_token else "#457b9d"
        for t in tokens
    ]

    # Reverse so highest prob is at the top
    tokens_r = tokens[::-1]
    probs_r = probs[::-1]
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


def render_sentence_with_highlight(history: list[tuple[str, float]]) -> str:
    """
    Render the generated sentence as HTML with each token coloured by
    confidence: green (high) → yellow → red (low).
    """
    if not history:
        return ""

    parts = []
    for token, prob in history:
        if prob > 0.5:
            colour = "#2dc653"   # green
        elif prob > 0.2:
            colour = "#f4a261"   # orange
        else:
            colour = "#e63946"   # red
        parts.append(
            f'<span style="color:{colour}; font-weight:bold;">{token}</span>'
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "sentence" not in st.session_state:
    st.session_state.sentence = ""
if "history" not in st.session_state:
    # list of (token_string, probability) for generated tokens
    st.session_state.history = []
if "started" not in st.session_state:
    st.session_state.started = False
if "top_tokens" not in st.session_state:
    st.session_state.top_tokens = []
if "top_probs" not in st.session_state:
    st.session_state.top_probs = []
if "selected_token" not in st.session_state:
    st.session_state.selected_token = ""
if "selected_prob" not in st.session_state:
    st.session_state.selected_prob = 0.0
if "entropy" not in st.session_state:
    st.session_state.entropy = 0.0
if "model_name" not in st.session_state:
    st.session_state.model_name = list(MODELS.keys())[0]

# ---------------------------------------------------------------------------
# Sidebar — model selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Model")
    chosen_label = st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
        index=0,
    )
    model_id = MODELS[chosen_label]

    if st.button("Load model"):
        st.session_state.model_name = chosen_label
        load_model(model_id)   # triggers cache
        st.success(f"Loaded: {chosen_label}")

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
        key="prompt_input",
    )

    btn_start, btn_next, btn_reset = st.columns(3)

    with btn_start:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            if prompt.strip():
                st.session_state.sentence = prompt.strip()
                st.session_state.history = []
                st.session_state.started = True
                tokenizer, model = load_model(model_id)
                (
                    st.session_state.top_tokens,
                    st.session_state.top_probs,
                    st.session_state.selected_token,
                    st.session_state.selected_prob,
                    st.session_state.entropy,
                ) = get_next_token_distribution(
                    tokenizer, model, st.session_state.sentence, top_k
                )

    with btn_next:
        if st.button("⏭ Next", use_container_width=True, disabled=not st.session_state.started):
            # Append selected token to sentence
            st.session_state.sentence += st.session_state.selected_token
            st.session_state.history.append(
                (st.session_state.selected_token, st.session_state.selected_prob)
            )
            tokenizer, model = load_model(model_id)
            (
                st.session_state.top_tokens,
                st.session_state.top_probs,
                st.session_state.selected_token,
                st.session_state.selected_prob,
                st.session_state.entropy,
            ) = get_next_token_distribution(
                tokenizer, model, st.session_state.sentence, top_k
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
        # Prompt in white, generated tokens colour-coded
        prompt_part = st.session_state.sentence[: len(prompt.strip()) if st.session_state.history else len(st.session_state.sentence)]
        generated_html = render_sentence_with_highlight(st.session_state.history)
        st.markdown(
            f'<p style="font-size:18px; line-height:1.8;">'
            f'{prompt_part}{generated_html}'
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
    else:
        st.info("Enter a prompt and click ▶ Start to begin.")
