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
# CSS — light theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

/* Light background */
.stApp { background-color: #f1f5f9; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }

/* All text dark */
p, li, span, label, div { color: #1e293b; }
h1, h2, h3, h4 { color: #0f172a !important; font-weight: 700 !important; }

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.15s ease;
    border: 1.5px solid #e2e8f0;
    color: #1e293b;
    background: #ffffff;
}
.stButton > button:hover {
    border-color: #6366f1;
    color: #6366f1;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(99,102,241,0.15);
}
.stButton > button[kind="primary"] {
    background: #6366f1;
    color: white !important;
    border: none;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
.stButton > button[kind="primary"]:hover {
    background: #4f46e5;
    color: white !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4);
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* Slider */
[data-testid="stSlider"] label { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stSlider"] > div > div > div > div { background: #6366f1 !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] p { color: #64748b !important; font-size: 0.72rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: #10b981 !important; }

/* Status box */
[data-testid="stStatus"] { border-radius: 10px; background: #f8fafc; }

/* Caption */
.stCaption p { color: #94a3b8 !important; font-size: 0.78rem !important; }

/* Success */
[data-testid="stAlert"][data-baseweb="notification"] { border-radius: 8px; }

/* Divider */
hr { border-color: #e2e8f0 !important; margin: 0.75rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "GPT-2 Medium (345M)":           "gpt2-medium",
    "GPT-2 Large (774M)":            "gpt2-large",
    "GPT-Neo 1.3B":                  "EleutherAI/gpt-neo-1.3B",
    "GPT-Neo 2.7B":                  "EleutherAI/gpt-neo-2.7B",
    "GPT-2 Small (117M)":            "gpt2",
    "DistilGPT-2 (82M)":             "distilgpt2",
    "Phi-2 (2.7B)":                  "microsoft/phi-2",
    "Phi-3-mini (3.8B)":             "microsoft/Phi-3-mini-4k-instruct",
    "Llama-3.2-3B (HF token)":       "meta-llama/Llama-3.2-3B",
    "Custom":                        "__custom__",
}

# ---------------------------------------------------------------------------
# Device
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
        st.write(f"Device: **{DEVICE.upper()}** · Precision: **{str(DTYPE).split('.')[-1]}**")
        st.write("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        st.write("Loading weights…")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE, device_map=DEVICE)
        st.write("Finalising…")
        model.eval()
        status.update(label="Model ready", state="complete", expanded=False)
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


def get_next_token_distribution(tokenizer, model, text, temperature=1.0, top_p=1.0):
    inputs = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    if temperature <= 0.01:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        selected_id = int(logits.argmax())
    else:
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            probs = _apply_top_p(probs, top_p)
        selected_id = int(torch.multinomial(probs, num_samples=1))

    k = min(15, int((probs > 0).sum()))
    top = torch.topk(probs, k=k)
    top_tokens = [tokenizer.decode([i]) for i in top.indices.tolist()]
    top_probs  = top.values.tolist()

    selected_token = tokenizer.decode([selected_id])
    selected_prob  = float(probs[selected_id])
    n_nucleus      = int((probs > 0).sum())
    entropy        = float(-torch.sum(probs * torch.log2(probs + 1e-12)))

    return top_tokens, top_probs, selected_token, selected_prob, entropy, n_nucleus

# ---------------------------------------------------------------------------
# Chart — vertical bars, full width
# ---------------------------------------------------------------------------

def make_chart(tokens, probs, selected_token) -> go.Figure:
    colours = [
        "#f97316" if t == selected_token else "#e0e7ff"
        for t in tokens
    ]
    border = [
        "#ea580c" if t == selected_token else "#c7d2fe"
        for t in tokens
    ]

    fig = go.Figure(go.Bar(
        x=[repr(t) for t in tokens],
        y=probs,
        marker=dict(color=colours, line=dict(color=border, width=1.5)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
        hovertemplate="<b>%{x}</b><br>%{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=11, color="#334155"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickformat=".0%",
            tickfont=dict(size=10, color="#94a3b8"),
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            range=[0, max(probs) * 1.3],
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter"),
        bargap=0.3,
    )
    return fig

# ---------------------------------------------------------------------------
# Sentence renderer
# ---------------------------------------------------------------------------

def render_sentence(prompt: str, history: list[tuple[str, float]]) -> str:
    parts = [f'<span style="color:#64748b;">{prompt}</span>']
    for token, prob in history:
        if prob > 0.5:
            colour, bg, border = "#065f46", "#d1fae5", "#a7f3d0"
        elif prob > 0.2:
            colour, bg, border = "#92400e", "#fef3c7", "#fde68a"
        else:
            colour, bg, border = "#991b1b", "#fee2e2", "#fecaca"
        parts.append(
            f'<span style="color:{colour}; background:{bg}; '
            f'border:1px solid {border}; border-radius:5px; '
            f'padding:1px 6px; margin:0 1px; font-weight:600; '
            f'font-size:1.15rem;">{token}</span>'
        )
    return "".join(parts)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = dict(
    sentence="", history=[], started=False, step=0,
    top_tokens=[], top_probs=[],
    selected_token="", selected_prob=0.0,
    entropy=0.0, n_nucleus=0,
    model_ready=False, model_label="",
    tokenizer=None, model=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar — model only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔍 Hesitate")
    st.caption("Step through LLM generation one token at a time.")
    st.divider()

    st.markdown("**Model**")
    chosen = st.selectbox(
        "model", list(MODELS.keys()), index=0,
        disabled=st.session_state.model_ready,
        label_visibility="collapsed",
    )
    model_id = MODELS[chosen]

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
            tok, mdl = load_model(model_id)
            st.session_state.tokenizer   = tok
            st.session_state.model       = mdl
            st.session_state.model_ready = True
            st.session_state.model_label = chosen
    else:
        st.success(f"✓ {st.session_state.model_label}")
        st.caption(f"`{DEVICE.upper()}` · `{str(DTYPE).split('.')[-1]}`")
        if st.button("⇄ Swap model", use_container_width=True):
            for k in ("model_ready", "started"):
                st.session_state[k] = False
            for k in ("tokenizer", "model"):
                st.session_state[k] = None
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem; color:#94a3b8; line-height:2;">'
        '🟢 &gt;50% — confident<br>'
        '🟡 20–50% — uncertain<br>'
        '🔴 &lt;20% — hesitating'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — top to bottom flow
# ---------------------------------------------------------------------------

# --- Header ---
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("## What is the model thinking?")
    st.caption(
        "Type a prompt → Start → step through generation one token at a time. "
        "The chart shows the full probability distribution before each pick."
    )
with c2:
    if st.session_state.started:
        st.markdown(
            f'<div style="text-align:right; padding-top:0.5rem;">'
            f'<span style="font-size:2.2rem; font-weight:700; color:#6366f1;">{st.session_state.step}</span>'
            f'<span style="color:#94a3b8; font-size:0.8rem; margin-left:4px;">tokens</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# --- Sentence hero ---
if st.session_state.sentence or not st.session_state.model_ready:
    sentence_html = (
        render_sentence(
            st.session_state.sentence[:len(st.session_state.sentence) - sum(len(t) for t, _ in st.session_state.history)],
            st.session_state.history
        )
        if st.session_state.history
        else f'<span style="color:#64748b; font-size:1.15rem;">{st.session_state.sentence or "Your sentence will grow here…"}</span>'
    )
else:
    sentence_html = '<span style="color:#cbd5e1; font-size:1.15rem;">Your sentence will grow here…</span>'

st.markdown(
    f'<div style="background:#ffffff; border:1.5px solid #e2e8f0; border-radius:14px; '
    f'padding:1.5rem 2rem; min-height:72px; line-height:2.2;">'
    f'{sentence_html}'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

# --- Prompt + Controls ---
pcol, bcol1, bcol2, bcol3 = st.columns([4, 1.2, 1.2, 0.5])

with pcol:
    prompt = st.text_input(
        "prompt",
        placeholder="The capital of France is",
        disabled=not st.session_state.model_ready,
        label_visibility="collapsed",
    )

with bcol1:
    start = st.button(
        "▶ Start",
        use_container_width=True,
        type="primary",
        disabled=not st.session_state.model_ready,
    )

with bcol2:
    nxt = st.button(
        "⏭ Next token",
        use_container_width=True,
        disabled=not (st.session_state.model_ready and st.session_state.started),
    )

with bcol3:
    rst = st.button("↺", use_container_width=True, help="Reset")

# --- Sampling controls ---
sc1, sc2, sc3 = st.columns([2, 2, 3])
with sc1:
    temperature = st.slider(
        "Temperature", 0.1, 2.0, 1.0, 0.05,
        help="Low = peaked distribution. High = flat distribution.",
    )
with sc2:
    top_p = st.slider(
        "Top-p (nucleus)", 0.1, 1.0, 1.0, 0.05,
        help="Truncate to tokens covering p% of probability mass.",
    )
with sc3:
    labels = {
        temperature <= 0.2: "🥶 Deterministic — always picks the top token",
        0.2 < temperature < 0.8: "❄️ Focused — top tokens dominate",
        0.8 <= temperature <= 1.2: "🌡️ Balanced — raw model output",
        1.2 < temperature <= 1.7: "🔥 Creative — surprising picks likely",
        temperature > 1.7: "🌋 Chaotic — almost random",
    }
    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    st.caption(next(v for k, v in labels.items() if k))
    if top_p < 1.0 and st.session_state.n_nucleus:
        st.caption(f"Nucleus: {st.session_state.n_nucleus} tokens covering {top_p*100:.0f}% of mass")

# --- Handle button actions ---
def _run():
    return get_next_token_distribution(
        st.session_state.tokenizer,
        st.session_state.model,
        st.session_state.sentence,
        temperature, top_p,
    )

if start and prompt.strip():
    st.session_state.sentence = prompt.strip()
    st.session_state.history  = []
    st.session_state.started  = True
    st.session_state.step     = 0
    (
        st.session_state.top_tokens, st.session_state.top_probs,
        st.session_state.selected_token, st.session_state.selected_prob,
        st.session_state.entropy, st.session_state.n_nucleus,
    ) = _run()
    st.rerun()

if nxt:
    st.session_state.sentence += st.session_state.selected_token
    st.session_state.history.append(
        (st.session_state.selected_token, st.session_state.selected_prob)
    )
    st.session_state.step += 1
    (
        st.session_state.top_tokens, st.session_state.top_probs,
        st.session_state.selected_token, st.session_state.selected_prob,
        st.session_state.entropy, st.session_state.n_nucleus,
    ) = _run()
    st.rerun()

if rst:
    for k, v in dict(
        sentence="", history=[], started=False, step=0,
        top_tokens=[], top_probs=[], selected_token="",
        selected_prob=0.0, entropy=0.0, n_nucleus=0,
    ).items():
        st.session_state[k] = v
    st.rerun()

st.divider()

# --- Metrics + Chart ---
if st.session_state.top_tokens:

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected token", repr(st.session_state.selected_token))
    m2.metric("Probability", f"{st.session_state.selected_prob*100:.1f}%")
    m3.metric("Entropy", f"{st.session_state.entropy:.2f} bits")
    m4.metric("Nucleus size", f"{st.session_state.n_nucleus:,}")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Chart title
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">'
        f'<span style="font-weight:600; color:#1e293b;">Next token distribution</span>'
        f'<span style="background:#fff7ed; color:#ea580c; border:1px solid #fed7aa; '
        f'border-radius:20px; padding:2px 12px; font-size:0.8rem; font-weight:600;">'
        f'{repr(st.session_state.selected_token)} · {st.session_state.selected_prob*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    fig = make_chart(
        st.session_state.top_tokens,
        st.session_state.top_probs,
        st.session_state.selected_token,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

elif st.session_state.model_ready:
    st.markdown(
        '<div style="background:#ffffff; border:1.5px solid #e2e8f0; border-radius:14px; '
        'padding:3rem; text-align:center; margin-top:1rem;">'
        '<div style="font-size:2rem; margin-bottom:0.5rem;">✍️</div>'
        '<div style="color:#94a3b8;">Type a prompt above and click Start</div>'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="background:#ffffff; border:1.5px solid #e2e8f0; border-radius:14px; '
        'padding:3rem; text-align:center; margin-top:1rem;">'
        '<div style="font-size:2rem; margin-bottom:0.5rem;">🧠</div>'
        '<div style="color:#94a3b8;">Load a model from the sidebar to get started</div>'
        '</div>',
        unsafe_allow_html=True,
    )
