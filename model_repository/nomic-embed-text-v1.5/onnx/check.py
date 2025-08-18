import onnx
import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path
from pprint import pprint

def load_model(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file {path} does not exist.")

    model = onnx.load(path)
    return model

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check.py <model.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]

    model = load_model(model_path)
    onnx.checker.check_model(model)

    print(f"Model IR Version: {model.ir_version}")
    print(f"Model Producer: {model.producer_name}")
    print(f"Model Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")

    model_input = model.graph.input
    model_output = model.graph.output
    model_initializers = model.graph.initializer

    print()
    print("Model Detailed:")
    print()

    print("[Model Input]")
    for x in model_input:
        tensor_type = x.type.tensor_type
        elem_type = tensor_type.elem_type
        dims = [dim.dim_param or dim.dim_value for dim in tensor_type.shape.dim]
        print("-", x.name, ":", onnx.TensorProto.DataType.Name(elem_type), dims)

    print()
    print("[Model Output]")
    for x in model_output:
        tensor_type = x.type.tensor_type
        elem_type = tensor_type.elem_type
        dims = [dim.dim_param or dim.dim_value for dim in tensor_type.shape.dim]
        print("-", x.name, ":", onnx.TensorProto.DataType.Name(elem_type), dims)


    print()
    print("Session Details:")
    print()

    session = ort.InferenceSession(model_path)

    session_metadata = session.get_modelmeta()
    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    print("[Session Metadata]")
    print("Producer:", session_metadata.producer_name)
    print("Domain:", session_metadata.domain)
    print("Version:", session_metadata.version)
    print("Description:", session_metadata.description)
    print("Graph Name:", session_metadata.graph_name)
    print("Graph Description:", session_metadata.graph_description)
    print("Custom Metadata:", session_metadata.custom_metadata_map)
    print()
    print("[Session Inputs]")
    for x in session_inputs:
        print("-", x.name, ":", x.type, x.shape)
    print()
    print("[Session Outputs]")
    for x in session_outputs:
        print("-", x.name, ":", x.type, x.shape)

