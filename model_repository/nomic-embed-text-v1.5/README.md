# nomic-embed text v1.5

## Download a Model

```bash
mkdir -p model_repository/nomic-embed-text-v1.5/1
cd model_repository/nomic-embed-text-v1.5/1
```

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "nomic-ai/nomic-embed-text-v1.5" --local-dir nomic-embed-text-v1.5
```

```bash
model_repository
└── nomic-embed-text-v1.5
    ├── 1
    │   └── nomic-embed-text-v1.5
    │       ├── 1_Pooling
    │       │   └── config.json
    │       ├── config.json
    │       ├── config_sentence_transformers.json
    │       ├── model.safetensors
    │       ├── modules.json
    │       ├── onnx
    │       │   ├── model_bnb4.onnx
    │       │   ├── model_fp16.onnx
    │       │   ├── model_int8.onnx
    │       │   ├── model.onnx
    │       │   ├── model_q4f16.onnx
    │       │   ├── model_q4.onnx
    │       │   ├── model_quantized.onnx
    │       │   └── model_uint8.onnx
    │       ├── README.md
    │       ├── sentence_bert_config.json
    │       ├── special_tokens_map.json
    │       ├── tokenizer_config.json
    │       ├── tokenizer.json
    │       └── vocab.txt
    └── config.pbtxt
```

## Run triton

in this project root:

```bash
docker run --rm \
--gpus 1 \
-p 8000:8000 \
-p 8001:8001 \
-p 8002:8002 \
-v ${PWD}/model_repository:/model_repository \
nvcr.io/nvidia/tritonserver:25.07-py3 \
tritonserver \
--model-repository /model_repository \
--model-control-mode explicit \
--load-model nomic-embed-text-v1.5 \
--log-verbose 1 \
--log-info 1 \
--log-error 1
```

client:

```bash
python model_repository/nomic-embed-text-v1.5/client/client.py \
--model-name nomic-embed-text-v1.5
```

### Error Message

```bash
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

Install Container Toolkit: [docs/docker](../../docs/docker.md)

## HTTP Client Request

```bash
cd model_repository/nomic-embed-text-v1.5/tests
```

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d @input.json \
  http://localhost:8000/v2/models/nomic-embed-text-v1.5/infer | jq | bat -l json
```

```json
{
  "model_name": "nomic-embed-text-v1.5",
  "model_version": "1",
  "outputs": [
    {
      "name": "last_hidden_state",
      "datatype": "FP32",
      "shape": [
        1,
        9,
        768
      ],
      "data": [
        0.1320551633834839,
        0.2668861746788025
      ]
    }
  ]
}
```

---

## ONNX Graph

- huggingface: [ONNX](https://huggingface.co/docs/transformers/serialization)
- github: [lutzroeder/netron](https://github.com/lutzroeder/netron)

### Packages

```bash
uv pip install 'optimum[exporters]'
# uv pip install onnxruntime-tools
uv pip install accelerate
```

### check ONNX Model

```bash
python onnx/check.py 1/nomic-embed-text-v1.5/onnx/model.onnx
```

```bash
Model IR Version: 7
Model Producer: pytorch
Model Domain:
Model Version: 0

Model Detailed:

[Model Input]
- input_ids : INT64 ['batch_size', 'sequence_length']
- token_type_ids : INT64 ['batch_size', 'sequence_length']
- attention_mask : INT64 ['batch_size', 'sequence_length']

[Model Output]
- last_hidden_state : FLOAT ['batch_size', 'sequence_length', 768]

Session Details:

[Session Metadata]
Producer: pytorch
Domain:
Version: 9223372036854775807
Description:
Graph Name: main_graph
Graph Description:
Custom Metadata: {}

[Session Inputs]
- input_ids : tensor(int64) ['batch_size', 'sequence_length']
- token_type_ids : tensor(int64) ['batch_size', 'sequence_length']
- attention_mask : tensor(int64) ['batch_size', 'sequence_length']

[Session Outputs]
- last_hidden_state : tensor(float) ['batch_size', 'sequence_length', 768]
```

### safetensors to onnx

- huggingface: [nomic-ai/nomic-bert-2048](https://huggingface.co/nomic-ai/nomic-bert-2048)
  - configuration_hf_nomic_bert.py
  - modeling_hf_nomic_bert.py

```bash
uv pip install einops
uv pip install sentence-transformers
```

#### feature-extraction, sentence-similarity

```bash
cd model_repository/nomic-embed-text-v1.5
```

```bash
optimum-cli export onnx \
--task feature-extraction \
--model ./1/nomic-embed-text-v1.5/ \
--trust-remote-code \
./1/nomic-embed-text-v1.5/feature-extraction/
```

or:

```bash
optimum-cli export onnx \
--task sentence-similarity \
--model ./1/nomic-embed-text-v1.5/ \
--trust-remote-code \
./1/nomic-embed-text-v1.5/sentence-similarity/
```

```bash
# some warnings are expected

 The exported model was saved at: 1/nomic-embed-text-v1.5/feature-extraction
```

### Check ONNX Model

#### feature-extraction

```bash
python onnx/check.py 1/nomic-embed-text-v1.5/feature-extraction/model.onnx
```

```bash
Model IR Version: 7
Model Producer: pytorch
Model Domain:
Model Version: 0

Model Detailed:

[Model Input]
- input_ids : INT64 ['batch_size', 'sequence_length']
- attention_mask : INT64 ['batch_size', 'sequence_length']

[Model Output]
- token_embeddings : FLOAT ['batch_size', 'sequence_length', 768]
- sentence_embedding : FLOAT ['batch_size', 'Concatsentence_embedding_dim_1']

Session Details:

[Session Metadata]
Producer: pytorch
Domain:
Version: 9223372036854775807
Description:
Graph Name: main_graph
Graph Description:
Custom Metadata: {}

[Session Inputs]
- input_ids : tensor(int64) ['batch_size', 'sequence_length']
- attention_mask : tensor(int64) ['batch_size', 'sequence_length']

[Session Outputs]
- token_embeddings : tensor(float) ['batch_size', 'sequence_length', 768]
- sentence_embedding : tensor(float) ['batch_size', 768]
```

#### sentence-similarity

```bash
python onnx/check.py 1/nomic-embed-text-v1.5/sentence-similarity/model.onnx
```

```bash
Model IR Version: 7
Model Producer: pytorch
Model Domain:
Model Version: 0

Model Detailed:

[Model Input]
- input_ids : INT64 ['batch_size', 'sequence_length']
- attention_mask : INT64 ['batch_size', 'sequence_length']

[Model Output]
- token_embeddings : FLOAT ['batch_size', 'sequence_length', 768]
- sentence_embedding : FLOAT ['batch_size', 'Concatsentence_embedding_dim_1']

Session Details:

[Session Metadata]
Producer: pytorch
Domain:
Version: 9223372036854775807
Description:
Graph Name: main_graph
Graph Description:
Custom Metadata: {}

[Session Inputs]
- input_ids : tensor(int64) ['batch_size', 'sequence_length']
- attention_mask : tensor(int64) ['batch_size', 'sequence_length']

[Session Outputs]
- token_embeddings : tensor(float) ['batch_size', 'sequence_length', 768]
- sentence_embedding : tensor(float) ['batch_size', 768]
```

