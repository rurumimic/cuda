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

## Download Models

### nomic-embed-text-v1.5

```bash
mkdir -p model_repository/nomic-embed-text-v1.5/1
cd model_repository/nomic-embed-text-v1.5/1
```

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "nomic-ai/nomic-embed-text-v1.5" --local-dir nomic-embed-text-v1.5
```

### gte-multilingual-reranker-base

```bash
mkdir -p model_repository/gte-multilingual-reranker-base/1
cd model_repository/gte-multilingual-reranker-base/1
```

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "Alibaba-NLP/gte-multilingual-reranker-base" --local-dir gte-multilingual-reranker-base
```

### roberta-base-go_emotions

```bash
mkdir -p model_repository/roberta-base-go_emotions/1
cd model_repository/roberta-base-go_emotions/1
```

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "SamLowe/roberta-base-go_emotions" --local-dir roberta-base-go_emotions
```

### efficient-splade-VI-BT-large-query

```bash
mkdir -p model_repository/efficient-splade-VI-BT-large-query/1
cd model_repository/efficient-splade-VI-BT-large-query/1
```

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "naver/efficient-splade-VI-BT-large-query" --local-dir efficient-splade-VI-BT-large-query
```

### Models

```bash
model_repository
├── efficient-splade-VI-BT-large-query
│   └── 1
│       └── efficient-splade-VI-BT-large-query
│           ├── config.json
│           ├── pytorch_model.bin
│           ├── README.md
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           └── vocab.txt
├── gte-multilingual-reranker-base
│   └── 1
│       └── gte-multilingual-reranker-base
│           ├── config.json
│           ├── images
│           ├── model.safetensors
│           ├── README.md
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── tokenizer.json
├── nomic-embed-text-v1.5
│   └── 1
│      └── nomic-embed-text-v1.5
│          ├── 1_Pooling
│          ├── config.json
│          ├── config_sentence_transformers.json
│          ├── model.safetensors
│          ├── modules.json
│          ├── onnx
│          ├── README.md
│          ├── sentence_bert_config.json
│          ├── special_tokens_map.json
│          ├── tokenizer_config.json
│          ├── tokenizer.json
│          └── vocab.txt
└── roberta-base-go_emotions
    └── 1
        └── roberta-base-go_emotions
            ├── config.json
            ├── merges.txt
            ├── model.safetensors
            ├── pytorch_model.bin
            ├── README.md
            ├── special_tokens_map.json
            ├── tokenizer_config.json
            ├── tokenizer.json
            ├── trainer_state.json
            └── vocab.json
```

