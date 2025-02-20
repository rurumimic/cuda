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
-v ${PWD}/model_repository:/models \
nvcr.io/nvidia/tritonserver:25.01-py3 \
tritonserver \
--model-repository /models \
--model-control-mode explicit \
--load-model nomic-embed-text-v1.5 \
--log-verbose 1 \
--log-info 1 \
--log-error 1
```

## HTTP Client Request

```bash
cd model_repository/nomic-embed-text-v1/tests
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

