import tensorrt

import logging

logger = logging.getLogger(__name__)

def test_version():
    logger.info(f"Torch-TensorRT version: {tensorrt.__version__}")

    # assert tensorrt.Builder(tensorrt.Logger())

