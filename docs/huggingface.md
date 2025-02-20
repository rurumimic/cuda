# HuggingFace

## Install HuggingFace CLI

### Python venv

```bash
uv venv
source .venv/bin/activate
```

### Install HF Transfer

```bash
uv pip install "huggingface_hub[hf_transfer]"
```

## Download a Model

```bash
mkdir -p model_repository/nomic-embed-text-v1.5/1
cd model_repository/nomic-embed-text-v1.5/1
```

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "nomic-ai/nomic-embed-text-v1.5" --local-dir nomic-embed-text-v1.5
```

