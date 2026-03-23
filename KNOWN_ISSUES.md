# Known Issues & Stack Limitations

A running log of technical problems encountered, their root cause, and what a
better stack might solve. Consult this before choosing a new stack or framework.

---

## 1. Model loading cannot be truly cancelled mid-download

**Status**: Partial workaround in place
**Affects**: Sidebar "Cancel" button during model load

### What happens
When the user clicks Cancel while a large model is downloading (e.g. gpt2-large,
774MB), the download continues in the background thread until completion. The UI
stops showing progress and discards the result, but the network transfer and disk
write cannot be interrupted.

### Root cause
`AutoModelForCausalLM.from_pretrained()` is a synchronous blocking call. Python
has no safe way to kill a running thread mid-execution. The cancel flag we set is
only checked *between* stages (after tokenizer, before model weights), not during
the download itself.

### What a better stack would give us
- **subprocess-based loading**: run the model load in a separate process instead
  of a thread. A process can be killed cleanly with `process.terminate()`.
- **huggingface_hub streaming**: use `hf_hub_download` with a custom progress
  callback that checks a cancel event at each chunk — gives true mid-download
  cancellation.
- **FastAPI + WebSocket backend**: move model loading entirely to the backend.
  The frontend sends a cancel signal over the WebSocket; the backend kills the
  subprocess. Cleaner separation of concerns.

---

## 2. Streamlit polling loop causes unnecessary re-renders

**Status**: Known limitation, no fix attempted
**Affects**: UI responsiveness during model loading

### What happens
To animate the progress bar, the main thread calls `time.sleep(0.5)` then
`st.rerun()` in a loop. Every 0.5s the entire Streamlit app re-renders from
scratch. This is wasteful and can cause flickering.

### Root cause
Streamlit has no native push/async mechanism. There is no way for a background
thread to update the UI without triggering a full page rerun.

### Additional note — GIL starvation
A tight `st.rerun()` loop with no sleep completely starves the background loading
thread. Python's Global Interpreter Lock (GIL) is never released by the spinning
main thread, so the background thread never runs. A `time.sleep(0.1)` call is
required to yield the GIL. This is a fundamental Python threading limitation.

### What a better stack would give us
- **FastAPI + React**: the backend emits Server-Sent Events (SSE) or WebSocket
  messages with progress updates. The frontend updates only the progress bar
  component — no full page re-render. No GIL issue since loading runs in a
  separate process, not a thread.

---

## 3. Model weights held in a global dict, not proper cache

**Status**: Works, but fragile
**Affects**: Multi-user scenarios, model swapping

### What happens
Model and tokenizer are stored in a module-level dict `_load_state`. This works
for a single user but would cause race conditions if multiple users loaded
different models simultaneously. Swapping models does not free the old model's
memory explicitly.

### Root cause
Streamlit's `st.cache_resource` would be the right tool, but it cannot be
invalidated programmatically when the user wants to swap models. We worked around
it with a plain dict, losing the memory management benefits.

### What a better stack would give us
- **FastAPI backend with explicit model lifecycle**: load/unload endpoints,
  proper `del model; torch.cuda.empty_cache()` on swap, one model per worker.

---

## 4. No GPU support

**Status**: Not implemented
**Affects**: Speed on machines with a GPU or Apple Silicon MPS

### What happens
Models always load with `torch_dtype=torch.float32` on CPU. On a MacBook with
Apple Silicon, MPS (Metal Performance Shaders) would give a significant speedup.

### What to add
Detect device at startup:
```python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model.to(device)
```
