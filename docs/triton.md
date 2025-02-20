# Triton

## Docker

- catalog: [tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)

```bash
docker pull nvcr.io/nvidia/tritonserver:25.01-py3
```

### Run triton

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

