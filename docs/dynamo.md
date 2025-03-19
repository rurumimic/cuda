# Dynamo

- github: [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)

## Build

### Clone a repository

```bash
git clone https://github.com/ai-dynamo/dynamo
cd dynamo
```

### Setup a virtual environment

```bash
uv venv
source .venv/bin/activate
uv pip install 'ai-dynamo[all]'
```

### Build with CUDA

```bash
cargo build --features cuda
```

### Run

```bash
target/debug/dynamo-run deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

```log
2025-03-19T15:34:14.158093Z  INFO dynamo_run: Using default engine: mistralrs. Use out=<engine> to specify one of echo_core, echo_full, mistralrs, sglang, vllm
2025-03-19T15:34:14.158133Z  INFO dynamo_run: CUDA on
2025-03-19T15:34:14.158467Z  INFO hf_hub: Using token file found "/.cache/huggingface/token"

# ...

2025-03-19T15:41:04.448962Z  INFO mistralrs_core::pipeline::normal: Prompt chunk size is 512.
2025-03-19T15:41:04.468213Z  INFO mistralrs_core::utils::normal: Detected minimum CUDA compute capability 8.6
2025-03-19T15:41:04.606504Z  INFO mistralrs_core::utils::normal: DType selected is BF16.
2025-03-19T15:41:04.606710Z  INFO mistralrs_core::utils::log: Automatic loader type determined to be `qwen2`
2025-03-19T15:41:04.713474Z  INFO mistralrs_core::pipeline::loaders: Using automatic device mapping parameters: text[max_seq_len: 4096, max_batch_size: 1].
2025-03-19T15:41:04.713646Z  INFO mistralrs_core::utils::log: Model has 28 repeating layers.
2025-03-19T15:41:04.713676Z  INFO mistralrs_core::utils::log: Loading model according to the following repeating layer mappings:
2025-03-19T15:41:04.713696Z  INFO mistralrs_core::utils::log: Layers 0-27: cuda[0]
2025-03-19T15:41:04.733485Z  INFO mistralrs_core::utils::normal: Detected minimum CUDA compute capability 8.6
2025-03-19T15:41:04.734116Z  INFO mistralrs_core::utils::normal: DType selected is BF16.
2025-03-19T15:41:04.734242Z  INFO mistralrs_core::pipeline::normal: Model config: Config { vocab_size: 151936, hidden_size: 1536, intermediate_size: 8960, num_hidden_layers: 28, num_attention_heads: 12, num_key_value_heads: 2, max_position_embeddings: 131072, sliding_window: 4096, rope_theta: 10000.0, rms_norm_eps: 1e-6, hidden_act: Silu, use_flash_attn: false, quantization_config: None, tie_word_embeddings: false }
2025-03-19T15:41:09.819697Z  INFO mistralrs_core::paged_attention: Allocating 112 MB for PagedAttention KV cache per GPU
2025-03-19T15:41:09.819738Z  INFO mistralrs_core::paged_attention: Using PagedAttention with block size 32 and 128 GPU blocks: available context length is 4096 tokens
2025-03-19T15:41:10.616974Z  INFO mistralrs_core::pipeline::chat_template: bos_toks = "<｜begin▁of▁sentence｜>", eos_toks = "<｜end▁of▁sentence｜>", unk_tok = `None`
2025-03-19T15:41:10.630020Z  INFO mistralrs_core: Enabling GEMM reduced precision in BF16.
2025-03-19T15:41:10.667186Z  INFO mistralrs_core: Enabling GEMM reduced precision in F16.
2025-03-19T15:41:10.668075Z  INFO mistralrs_core::cublaslt: Initialized cuBLASlt handle
2025-03-19T15:41:10.668240Z  INFO mistralrs_core: Beginning dummy run.
2025-03-19T15:41:18.246273Z  INFO mistralrs_core: Dummy run completed in 7.577981889s.
2025-03-19T15:41:18.246366Z  INFO dynamo_run::input::text: Ctrl-c to exit
```

```console
✔ User · hello?

Alright, someone greeted me with "hello?" and I replied "Hello! How can I assist you today?"
They must be new or just starting out. I should probably let them know I'm here to help with whatever they need.
Maybe I can offer some basic information or ask them what they're looking for.
</think>

Hello! How can I assist you today?
```

