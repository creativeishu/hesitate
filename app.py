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
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background-color: #0d1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f9fafb;
}

/* Slider accent */
[data-testid="stSlider"] > div > div > div > div {
    background: #6366f1;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    border: none;
    transition: all 0.2s ease;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
}

/* Section headers */
h2 { color: #f9fafb !important; font-weight: 600 !important; }
h3 { color: #e5e7eb !important; font-weight: 500 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #f9fafb !important; font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #10b981 !important; }

/* Info/success boxes */
[data-testid="stAlert"] { border-radius: 10px; }

/* Divider */
hr { border-color: #1f2937 !important; }

/* Caption text */
.stCaption { color: #6b7280 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

MODELS = {
    "GPT-2 Medium (345M)":               "gpt2-medium",
    "GPT-2 Large (774M)":                "gpt2-large",
    "GPT-Neo 1.3B":                      "EleutherAI/gpt-neo-1.3B",
    "GPT-Neo 2.7B":                      "EleutherAI/gpt-neo-2.7B",
    "GPT-2 (small, 117M)":               "gpt2",
    "DistilGPT-2 (82M)":                 "distilgpt2",
    "Phi-2 (2.7B) — instruct":           "microsoft/phi-2",
    "Phi-3-mini (3.8B) — instruct":      "microsoft/Phi-3-mini-4k-instruct",
    "Llama-3.2-3B (needs HF token)":     "meta-llama/Llama-3.2-3B",
    "Custom (type below)":               "__custom__",
}

# ---------------------------------------------------------------------------
# Device & dtype
# ---------------------------------------------------------------------------

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)
DTYPE = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    with st.status("Loading model…", expanded=True) as status:
        st.write(f"Device: **{DEVICE.upper()}** | Precision: **{str(DTYPE).split('.')[-1]}**")
        st.write("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        st.write("Loading model weights…")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=DTYPE, device_map=DEVICE
        )
        st.write("Finalising…")
        model.eval()
        status.update(label="✓ Model ready", state="complete", expanded=False)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    remove = (cumulative - sorted_probs) >= top_p
    sorted_probs[remove] = 0.0
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx] = sorted_probs
    return filtered / filtered.sum()


def get_next_token_distribution(
    tokenizer, model, text: str, temperature: float = 1.0, top_p: float = 1.0
):
    inputs = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    if temperature <= 0.01:
        probs_all = torch.zeros_like(logits)
        probs_all[logits.argmax()] = 1.0
        selected_id = int(logits.argmax())
    else:
        probs_all = torch.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            probs_all = _apply_top_p(probs_all, top_p)
        selected_id = int(torch.multinomial(probs_all, num_samples=1))

    k = min(20, int((probs_all > 0).sum()))
    top = torch.topk(probs_all, k=k)
    top_tokens = [tokenizer.decode([tid]) for tid in top.indices.tolist()]
    top_probs  = top.values.tolist()

    selected_token = tokenizer.decode([selected_id])
    selected_prob  = float(probs_all[selected_id])
    n_nucleus      = int((probs_all > 0).sum())
    entropy_bits   = float(-torch.sum(probs_all * torch.log2(probs_all + 1e-12)))

    return top_tokens, top_probs, selected_token, selected_prob, entropy_bits, n_nucleus


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLOUR_SELECTED = "#f97316"   # warm orange
COLOUR_NUCLEUS  = "#6366f1"   # indigo
COLOUR_TAIL     = "#1e3a5f"   # muted blue


def make_histogram(tokens, probs, selected_token) -> go.Figure:
    colours = [COLOUR_SELECTED if t == selected_token else COLOUR_NUCLEUS for t in tokens]
    tokens_r  = tokens[::-1]
    probs_r   = probs[::-1]
    colours_r = colours[::-1]

    fig = go.Figure(go.Bar(
        x=probs_r,
        y=[repr(t) for t in tokens_r],
        orientation="h",
        marker=dict(
            color=colours_r,
            line=dict(width=0),
        ),
        text=[f"{p*100:.1f}%" for p in probs_r],
        textposition="outside",
        textfont=dict(size=12, color="#e5e7eb"),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(
            range=[0, max(probs) * 1.3],
            showgrid=True,
            gridcolor="#1f2937",
            tickformat=".0%",
            tickfont=dict(color="#6b7280", size=11),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#e5e7eb", size=12),
            ticklabelposition="outside left",
        ),
        margin=dict(l=10, r=70, t=10, b=10),
        height=540,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(family="Inter", color="#e5e7eb"),
        bargap=0.25,
    )
    return fig


def render_sentence(prompt: str, history: list[tuple[str, float]]) -> str:
    parts = [f'<span style="color:#9ca3af;">{prompt}</span>']
    for token, prob in history:
        if prob > 0.5:
            colour, bg = "#10b981", "#052e16"
        elif prob > 0.2:
            colour, bg = "#f59e0b", "#2d1f00"
        else:
            colour, bg = "#f87171", "#2d0a0a"
        parts.append(
            f'<span style="color:{colour}; background:{bg}; '
            f'border-radius:4px; padding:2px 5px; margin:0 1px; '
            f'font-weight:600;">{token}</span>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = dict(
    sentence="", history=[], started=False,
    top_tokens=[], top_probs=[],
    selected_token="", selected_prob=0.0,
    entropy=0.0, n_nucleus=0, step=0,
    model_ready=False, model_label="",
    tokenizer=None, model=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 Hesitate")
    st.caption("Slow down generation. See the model think.")
    st.divider()

    st.markdown("#### Model")
    chosen_label = st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
        index=0,
        disabled=st.session_state.model_ready,
        label_visibility="collapsed",
    )
    model_id = MODELS[chosen_label]

    if model_id == "__custom__":
        model_id = st.text_input(
            "HuggingFace model ID",
            placeholder="e.g. EleutherAI/gpt-neo-1.3B",
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
            st.session_state.tokenizer   = tokenizer
            st.session_state.model       = model
            st.session_state.model_ready = True
            st.session_state.model_label = chosen_label

    if st.session_state.model_ready:
        st.success(f"✓ {st.session_state.model_label}")
        st.caption(f"Device: `{DEVICE.upper()}` · Precision: `{str(DTYPE).split('.')[-1]}`")
        if st.button("⇄ Swap model", use_container_width=True):
            for k in ("model_ready", "started"):
                st.session_state[k] = False
            for k in ("tokenizer", "model"):
                st.session_state[k] = None
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown("#### Sampling")

    temperature = st.slider(
        "Temperature",
        min_value=0.1, max_value=2.0, value=1.0, step=0.05,
        help="Scales logits before softmax. Low = peaked, high = flat.",
    )
    t_label = (
        "🥶 Deterministic" if temperature <= 0.2 else
        "❄️ Focused"       if temperature <  0.8 else
        "🌡️ Balanced"      if temperature <= 1.2 else
        "🔥 Creative"      if temperature <= 1.7 else
        "🌋 Chaotic"
    )
    st.caption(f"{t_label} — T = {temperature:.2f}")

    top_p = st.slider(
        "Top-p (nucleus)",
        min_value=0.1, max_value=1.0, value=1.0, step=0.05,
        help="Keep smallest set of tokens covering p% of probability mass.",
    )
    if top_p < 1.0:
        nucleus_info = f"Nucleus: {top_p*100:.0f}% of mass"
        if st.session_state.n_nucleus:
            nucleus_info += f" · {st.session_state.n_nucleus} tokens"
        st.caption(nucleus_info)
    else:
        st.caption("Full vocabulary · no nucleus truncation")

    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem; color:#4b5563; line-height:1.8;">'
        '🟢 &nbsp;> 50% confident<br>'
        '🟠 &nbsp;20–50% uncertain<br>'
        '🔴 &nbsp;< 20% hesitating'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_step = st.columns([3, 1])
with col_title:
    st.markdown("## Token-by-token generation")
    st.caption(
        "Enter a prompt, hit **Start**, then step through generation one token at a time. "
        "The histogram shows the model's probability distribution before each choice."
    )
with col_step:
    if st.session_state.started:
        st.markdown(
            f'<div style="text-align:right; padding-top:1rem;">'
            f'<span style="font-size:2rem; font-weight:700; color:#6366f1;">'
            f'Step {st.session_state.step}</span><br>'
            f'<span style="color:#6b7280; font-size:0.8rem;">tokens generated</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

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

    # Prompt input
    prompt = st.text_area(
        "Prompt",
        placeholder="The capital of France is",
        height=90,
        disabled=not st.session_state.model_ready,
        label_visibility="collapsed",
    )

    # Controls
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
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
                st.session_state.step     = 0
                (
                    st.session_state.top_tokens,
                    st.session_state.top_probs,
                    st.session_state.selected_token,
                    st.session_state.selected_prob,
                    st.session_state.entropy,
                    st.session_state.n_nucleus,
                ) = _run_inference()

    with b2:
        if st.button(
            "⏭ Next token",
            use_container_width=True,
            disabled=not (st.session_state.model_ready and st.session_state.started),
        ):
            st.session_state.sentence += st.session_state.selected_token
            st.session_state.history.append(
                (st.session_state.selected_token, st.session_state.selected_prob)
            )
            st.session_state.step += 1
            (
                st.session_state.top_tokens,
                st.session_state.top_probs,
                st.session_state.selected_token,
                st.session_state.selected_prob,
                st.session_state.entropy,
                st.session_state.n_nucleus,
            ) = _run_inference()

    with b3:
        if st.button("↺", use_container_width=True, help="Reset"):
            st.session_state.sentence       = ""
            st.session_state.history        = []
            st.session_state.started        = False
            st.session_state.step           = 0
            st.session_state.top_tokens     = []
            st.session_state.top_probs      = []
            st.session_state.selected_token = ""
            st.session_state.selected_prob  = 0.0
            st.session_state.entropy        = 0.0
            st.session_state.n_nucleus      = 0

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Sentence display
    if st.session_state.sentence:
        sentence_html = render_sentence(prompt.strip(), st.session_state.history)
    else:
        sentence_html = '<span style="color:#374151;">Your sentence will appear here…</span>'

    st.markdown(
        f'<div style="'
        f'background:#111827; border:1px solid #1f2937; border-radius:12px; '
        f'padding:1.5rem 1.8rem; min-height:90px; '
        f'font-size:1.25rem; line-height:2.2; letter-spacing:0.01em;">'
        f'{sentence_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metrics
    if st.session_state.started:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Next token",
            repr(st.session_state.selected_token),
            f"{st.session_state.selected_prob*100:.1f}%",
        )
        m2.metric(
            "Entropy",
            f"{st.session_state.entropy:.2f}",
            help="Bits of uncertainty. Higher = model is more spread out.",
        )
        m3.metric(
            "Nucleus",
            f"{st.session_state.n_nucleus:,}",
            help="Tokens with non-zero probability after top-p truncation.",
        )

with col_right:

    if st.session_state.top_tokens:
        # Chart label
        selected_repr = repr(st.session_state.selected_token)
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">'
            f'<span style="color:#9ca3af; font-size:0.85rem;">Next token distribution</span>'
            f'<span style="background:#f97316; color:white; border-radius:6px; '
            f'padding:2px 10px; font-size:0.85rem; font-weight:600;">'
            f'{selected_repr} selected</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        fig = make_histogram(
            st.session_state.top_tokens,
            st.session_state.top_probs,
            st.session_state.selected_token,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    elif not st.session_state.model_ready:
        st.markdown(
            '<div style="background:#111827; border:1px solid #1f2937; border-radius:12px; '
            'padding:3rem; text-align:center; margin-top:2rem;">'
            '<div style="font-size:2.5rem; margin-bottom:1rem;">🧠</div>'
            '<div style="color:#6b7280; font-size:1rem;">Load a model from the sidebar to begin</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#111827; border:1px solid #1f2937; border-radius:12px; '
            'padding:3rem; text-align:center; margin-top:2rem;">'
            '<div style="font-size:2.5rem; margin-bottom:1rem;">✍️</div>'
            '<div style="color:#6b7280; font-size:1rem;">Enter a prompt and click Start</div>'
            '</div>',
            unsafe_allow_html=True,
        )
