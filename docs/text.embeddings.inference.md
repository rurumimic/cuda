# Text Embeddings Inference

- github: [huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)

## Usage

### Docker

#### Docker Images

- readme: [docker-images](https://github.com/huggingface/text-embeddings-inference#docker-images)
- [packages](https://github.com/huggingface/text-embeddings-inference/pkgs/container/text-embeddings-inference)

| Architecture                        | Image                                                                   |
|-------------------------------------|-------------------------------------------------------------------------|
| CPU                                 | ghcr.io/huggingface/text-embeddings-inference:cpu-1.6                   |
| Volta                               | NOT SUPPORTED                                                           |
| Turing (T4, RTX 2000 series, ...)   | ghcr.io/huggingface/text-embeddings-inference:turing-1.6 (experimental) |
| Ampere 80 (A100, A30)               | ghcr.io/huggingface/text-embeddings-inference:1.6                       |
| Ampere 86 (A10, A40, ...)           | ghcr.io/huggingface/text-embeddings-inference:86-1.6                    |
| Ada Lovelace (RTX 4000 series, ...) | ghcr.io/huggingface/text-embeddings-inference:89-1.6                    |
| Hopper (H100)                       | ghcr.io/huggingface/text-embeddings-inference:hopper-1.6 (experimental) |

### Build from source

clone a repo:

```bash
git clone https://github.com/huggingface/text-embeddings-inference
# or
gh repo clone huggingface/text-embeddings-inference
```

```bash
cd text-embeddings-inference
```

#### Install packages

##### openssl library

```bash
# ubuntu
sudo apt-get install libssl-dev
```

check:

```bash
PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --libs --cflags openssl

-I/usr/include -lssl -lcrypto
```

#### Build router

```bash
# On x86
cargo build --release --bin text-embeddings-router -F mkl

# On M1 or M2
cargo build --release --bin text-embeddings-router -F metal

# Cuda
## On Turing GPUs (T4, RTX 2000 series ... )
export PATH=$PATH:/usr/local/cuda/bin
cargo build --release --bin text-embeddings-router -F candle-cuda-turing -F http --no-default-features
```

#### Binary

```bash
[drwxrwxr-x]  target/release
├── [drwxrwxr-x]  build
├── [drwxrwxr-x]  deps
├── [drwxrwxr-x]  examples
├── [drwxrwxr-x]  incremental
├── [-rwxrwxr-x]  text-embeddings-router
└── [-rw-rw-r--]  text-embeddings-router.d
```

```bash
export PATH="$PATH:$PWD/target/release"
```

```bash
text-embeddings-router -h
```

## Run

```bash
# embed
model=./model_repository/nomic-embed-text-v1.5/1/nomic-embed-text-v1.5

# rerank
model=./model_repository/gte-multilingual-reranker-base/1/gte-multilingual-reranker-base

# sequence classification
model=./model_repository/roberta-base-go_emotions/1/roberta-base-go_emotions

# splade pooling
model=./model_repository/efficient-splade-VI-BT-large-query/1/efficient-splade-VI-BT-large-query
```

### Run a server

```bash
text-embeddings-router --model-id $model --port 8080
```

mkl:

```log
2025-02-25T11:40:08.438257Z  INFO text_embeddings_router: router/src/main.rs:175: Args { model_id: "./mod**_**********/*****-*****-****-**.*/*/*****-*****-****-v1.5", revision: None, tokenization_workers: None, dtype: None, pooling: None, max_concurrent_requests: 512, max_batch_tokens: 16384, max_batch_requests: None, max_client_batch_size: 32, auto_truncate: false, default_prompt_name: None, default_prompt: None, hf_api_token: None, hostname: "0.0.0.0", port: 8080, uds_path: "/tmp/text-embeddings-inference-server", huggingface_hub_cache: None, payload_limit: 2000000, api_key: None, json_output: false, otlp_endpoint: None, otlp_service_name: "text-embeddings-inference.server", cors_allow_origin: None }
2025-02-25T11:40:08.454354Z  INFO text_embeddings_router: router/src/lib.rs:188: Maximum number of tokens per request: 8192
2025-02-25T11:40:08.454631Z  INFO text_embeddings_core::tokenization: core/src/tokenization.rs:28: Starting 12 tokenization workers
2025-02-25T11:40:08.502609Z  INFO text_embeddings_router: router/src/lib.rs:230: Starting model backend
2025-02-25T11:40:08.504046Z  INFO text_embeddings_backend_candle: backends/candle/src/lib.rs:217: Starting NomicBert model on Cpu
2025-02-25T11:40:09.109595Z  WARN text_embeddings_router: router/src/lib.rs:258: Backend does not support a batch size > 4
2025-02-25T11:40:09.109615Z  WARN text_embeddings_router: router/src/lib.rs:259: forcing `max_batch_requests=4`
2025-02-25T11:40:09.111174Z  INFO text_embeddings_router::http::server: router/src/http/server.rs:1812: Starting HTTP server: 0.0.0.0:8080
2025-02-25T11:40:09.111182Z  INFO text_embeddings_router::http::server: router/src/http/server.rs:1813: Ready
```

cuda:

```log
2025-02-25T12:14:30.575208Z  INFO text_embeddings_router: router/src/main.rs:175: Args { model_id: "./mod**_**********/*****-*****-****-**.*/*/*****-*****-****-v1.5", revision: None, tokenization_workers: None, dtype: None, pooling: None, max_concurrent_requests: 512, max_batch_tokens: 16384, max_batch_requests: None, max_client_batch_size: 32, auto_truncate: false, default_prompt_name: None, default_prompt: None, hf_api_token: None, hostname: "0.0.0.0", port: 8080, uds_path: "/tmp/text-embeddings-inference-server", huggingface_hub_cache: None, payload_limit: 2000000, api_key: None, json_output: false, otlp_endpoint: None, otlp_service_name: "text-embeddings-inference.server", cors_allow_origin: None }
2025-02-25T12:14:30.583439Z  INFO text_embeddings_router: router/src/lib.rs:188: Maximum number of tokens per request: 8192
2025-02-25T12:14:30.583597Z  INFO text_embeddings_core::tokenization: core/src/tokenization.rs:28: Starting 12 tokenization workers
2025-02-25T12:14:30.631476Z  INFO text_embeddings_router: router/src/lib.rs:230: Starting model backend
2025-02-25T12:14:31.078473Z  INFO text_embeddings_backend_candle: backends/candle/src/lib.rs:337: Starting NomicBert model on Cuda(CudaDevice(DeviceId(1)))
2025-02-25T12:14:37.676633Z  INFO text_embeddings_router::http::server: router/src/http/server.rs:1812: Starting HTTP server: 0.0.0.0:8080
2025-02-25T12:14:37.676648Z  INFO text_embeddings_router::http::server: router/src/http/server.rs:1813: Ready
```

### client

#### /embed

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

#### /rerank

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

#### /predict

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

#### /embed_sparse

```bash
text-embeddings-router --model-id $model --port 8080 --pooling splade
```

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

