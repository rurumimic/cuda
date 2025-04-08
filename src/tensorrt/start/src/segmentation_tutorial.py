#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import matplotlib.pyplot as plt
from PIL import Image

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "models/fcn-resnet101.engine"
input_file  = "images/input.ppm"
output_file = "images/output.ppm"

# For torchvision models, input images are loaded in to a range of [0, 1] and
# normalized using mean = [0.485, 0.456, 0.406] and stddev = [0.229, 0.224, 0.225].
def preprocess(image):
    # Mean normalization
    mean = np.array([0.485, 0.456, 0.406]).astype('float32')
    stddev = np.array([0.229, 0.224, 0.225]).astype('float32')
    data = (np.asarray(image).astype('float32') / float(255.0) - mean) / stddev
    # Switch from HWC to to CHW order
    return np.moveaxis(data, 2, 0)

def postprocess(data):
    num_classes = 21
    # create a color palette, selecting a color for each class
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette*i%255 for i in range(num_classes)]).astype("uint8")
    # plot the segmentation predictions for 21 classes in different colors
    data = np.clip(data, 0, 255).astype('uint8')
    img = Image.fromarray(data, mode='P')
    img.putpalette(colors)
    return img

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, input_file, output_file):
    print("Reading input image from file {}".format(input_file))
    with Image.open(input_file) as img:
        input_image = preprocess(img)
        image_width = img.width
        image_height = img.height

    with engine.create_execution_context() as context:
        stream = cuda.Stream()

        # Set the input shape
        input_name = [name for name in engine if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT][0]
        context.set_input_shape(input_name, (1, 3, image_height, image_width))

        # Make sure shape inference is done before allocating buffers
        context.infer_shapes()

        # Allocate memory
        buffers = {}

        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            shape = context.get_tensor_shape(tensor)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))
            size = trt.volume(shape)

            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                input_buffer = np.ascontiguousarray(input_image.astype(dtype))
                input_memory = cuda.mem_alloc(input_image.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
                buffers[tensor] = {
                        "host": input_buffer,
                        "device": input_memory,
                        "shape": shape,
                        "dtype": dtype,
                }
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))
                buffers[tensor] = {
                        "host": output_buffer,
                        "device": output_memory,
                        "shape": shape,
                        "dtype": dtype,
                }

        # Transfer input data to the GPU.
        for name, buffer in buffers.items():
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cuda.memcpy_htod_async(int(buffer["device"]), buffer["host"], stream)

        # Run inference
        context.execute_async_v3(stream_handle=stream.handle)

        # Transfer prediction output from the GPU.
        for name, buffer in buffers.items():
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(buffer["host"], int(buffer["device"]), stream)

        # Synchronize the stream
        stream.synchronize()

        output_names = [name for name in engine if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        output_name = output_names[0]

        raw_output = buffers[output_name]["host"]
        out_shape = buffers[output_name]["shape"]

        output_array = raw_output.reshape(out_shape)
        if out_shape[:2] == (1, 1):
            class_map = output_array[0, 0]
        else:
            raise RuntimeError("Unexpected output shape: {}".format(out_shape))

        np.savetxt('test.out', class_map.astype(np.int64), fmt='%i', delimiter=' ', newline=' ')


    img = postprocess(class_map)
    print("Writing output image to file {}".format(output_file))
    img.convert('RGB').save(output_file, "PPM")

if __name__ == "__main__":
    print("Running TensorRT inference for FCN-ResNet101")
    with load_engine(engine_file) as engine:
        infer(engine, input_file, output_file)

