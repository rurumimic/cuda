# Ensemble

## Run triton

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
--load-model ensemble \
--log-verbose 1 \
--log-info 1 \
--log-error 1
```

### Client

```bash
python model_repository/ensemble/client/client.py
```

