# DocProcessorPro local model servers

Two standalone FastAPI processes that DocProcessorPro's `_llm_client.py` talks
to over HTTP. Neither is part of the shipped GUI app — run them on whichever
machine hosts the models (Apple Silicon, via MLX-LM).

## Install

```bash
pip install -e ".[llm-server]"
```

## Run (mock mode — no model weights required)

Useful for developing/testing the rest of the pipeline before the small model
is chosen or Qwen3.6 weights are downloaded:

```bash
python -m llm_server.small_model_server --mock
python -m llm_server.large_model_server --mock
```

Both default to `127.0.0.1` on their respective ports (`8765` small, `8766`
large) — matching `_llm_client.py`'s defaults, so no App Settings changes are
needed to test against them.

## Run (real models)

```bash
python -m llm_server.small_model_server --model-path <small-model-path-or-repo-id> --embed-model-path <embedding-model-path-or-repo-id>
python -m llm_server.large_model_server --model-path <qwen3.6-27b-4bit-path-or-repo-id>
```

`--model-path` accepts anything `mlx_lm.load()` accepts (a local path or a
Hugging Face repo id).

The small server's `--embed-model-path` loads a *separate* dedicated
embedding model rather than pooling hidden states out of the generative
model — see `small_model_server._load_embedder()`. That function is a
placeholder until the small model is chosen; adjust it to match whatever
embedding runtime that model needs.

## Endpoints

Small model server (port 8765):
- `GET /health`
- `POST /extract_page_fields` — `{"text", "extraction_method"}` → `{"output": "<pipe-delimited line>"}` (see `../llm_extraction_schema.md`)
- `POST /embed` — `{"texts": [...]}` → `{"embeddings": [[...], ...]}`

Large model server (port 8766):
- `GET /health`
- `POST /summarize_case` — `{"case_payload": {...}}` → `{"output": "<json string>"}`
