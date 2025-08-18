import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tokenizers import Tokenizer
import sys
from pprint import pprint
from pathlib import Path
import os
import argparse

host = "localhost"
port = 8000
model_name = "ensemble"

def load_tokenizer(path, max_length=256):
    tokenizer = Tokenizer.from_file(path)
    tokenizer.enable_padding(length=max_length, pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer

def tokenize(tokenizer, texts):
    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer.encode_batch(texts)
    input_ids = np.asarray([enc.ids for enc in encoded], dtype=np.int64)
    attention_mask = np.asarray([enc.attention_mask for enc in encoded], dtype=np.int64)
    token_type_ids = np.asarray([enc.type_ids for enc in encoded], dtype=np.int64)
    return input_ids, attention_mask, token_type_ids



def main():
    args = argparse.ArgumentParser()
    args.add_argument("--host", type=str, default=host)
    args.add_argument("--port", type=int, default=port)
    args.add_argument("--tokenizer-path", type=str, default="tokenizer.json")
    args.add_argument("--texts", nargs="+", default=["Hello, world!", "This is a test."])
    args.add_argument("--max-length", type=int, default=256)
    args.add_argument("--verbose", action="store_true", default=False)
    args.add_argument("--load-model", action="store_true", default=False)
    args = args.parse_args()
    print("Triton Client")

    try:
        client = httpclient.InferenceServerClient(url=f"{args.host}:{args.port}", verbose=args.verbose)
    except Exception as e:
        print(f"context creation failed: {e}")
        return

    tokenizer_path = args.tokenizer_path
    max_length = args.max_length
    texts = args.texts

    tokenizer_path = Path("model_repository/nomic-embed-text-v1.5/1/nomic-embed-text-v1.5", tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Tokenizer file {tokenizer_path} does not exist.")
        return

    tokenizer = load_tokenizer(str(tokenizer_path), max_length=max_length)
    input_ids, attention_mask, token_type_ids = tokenize(tokenizer, texts)

    inputs = []
    inputs.append(httpclient.InferInput("input_ids", list(input_ids.shape), "INT64"))
    inputs.append(httpclient.InferInput("attention_mask", list(attention_mask.shape), "INT64"))
    inputs.append(httpclient.InferInput("token_type_ids", list(token_type_ids.shape), "INT64"))

    inputs[0].set_data_from_numpy(np.array(input_ids), binary_data=True)
    inputs[1].set_data_from_numpy(np.array(attention_mask), binary_data=True)
    inputs[2].set_data_from_numpy(np.array(token_type_ids), binary_data=True)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("last_hidden_state", binary_data=True))

    result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    resp = result.get_response()

    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"token_type_ids: {token_type_ids.shape}")
    print("input_ids:", input_ids)
    print("attention_mask:", attention_mask)
    print("token_type_ids:", token_type_ids)
    print(f"Response: {resp}")

    output_data = result.as_numpy("last_hidden_state")
    print(f"Output shape: {output_data.shape}")

    for i, text in enumerate(texts):
        print()
        print("Text:", text)
        print("Embedding shape:", end=" ")
        pprint(output_data[i].shape)
        pprint(output_data[i])


if __name__ == "__main__":
    main()
