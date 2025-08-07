# Triton

- docs: [user guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/contents.html)

### Image

- catalog: [tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)

```bash
docker pull nvcr.io/nvidia/tritonserver:25.07-py3
docker pull nvcr.io/nvidia/tritonserver:25.07-py3-igpu
docker pull nvcr.io/nvidia/tritonserver:25.07-py3-igpu-min
docker pull nvcr.io/nvidia/tritonserver:25.07-py3-igpu-sdk
docker pull nvcr.io/nvidia/tritonserver:25.07-py3-min
docker pull nvcr.io/nvidia/tritonserver:25.07-py3-sdk
docker pull nvcr.io/nvidia/tritonserver:25.07-pyt-python-py3
docker pull nvcr.io/nvidia/tritonserver:25.07-trtllm-python-py3
docker pull nvcr.io/nvidia/tritonserver:25.07-vllm-python-py3
```

## Examples

- [model_repository/nomic-embed-text-v1.5](../model_repository/nomic-embed-text-v1.5/README.md)

---

## Client

- github: [triton-inference-server/client](https://github.com/triton-inference-server/client)

```bash
pip install tritonclient[all]
```

