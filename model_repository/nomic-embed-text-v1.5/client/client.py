import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

host = "localhost"
port = 8000
model_name = "nomic-embed-text-v1.5"


def main():
    print("Triton Client")

    try:
        client = httpclient.InferenceServerClient(url=f"{host}:{port}", verbose=True)
    except Exception as e:
        print(f"context creation failed: {e}")
        return

    client.load_model(model_name)
    if not client.is_model_ready(model_name):
        print(f"Model {model_name} is not ready.")
        return
    
    input_ids = [[101, 2023, 2003, 1037, 7099, 3793, 2005, 7861, 1012]]
    attention_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
    token_type_ids = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]

    inputs = []
    inputs.append(httpclient.InferInput("input_ids", [1, 9], "INT64"))
    inputs.append(httpclient.InferInput("attention_mask", [1, 9], "INT64"))
    inputs.append(httpclient.InferInput("token_type_ids", [1, 9], "INT64"))

    inputs[0].set_data_from_numpy(np.array(input_ids), binary_data=True)
    inputs[1].set_data_from_numpy(np.array(attention_mask), binary_data=True)
    inputs[2].set_data_from_numpy(np.array(token_type_ids), binary_data=True)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("last_hidden_state", binary_data=True))

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    resp = result.get_response()
    print(f"Response: {resp}")

    output_data = result.as_numpy("last_hidden_state")
    print(f"Output length: {len(output_data)}")
    print(f"Output shape: {output_data.shape}")
    print(f"Output[0] length: {len(output_data[0])}")
    print(f"Output[0] shape: {output_data[0].shape}")
    print("Output", output_data[0])
    # for i in range(9):
    #     print(f"Output {i}: {output_data[0][i]}")


if __name__ == "__main__":
    main()
