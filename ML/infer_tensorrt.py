import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def convert_to_tensorrt(onnx_file_path):
    trt_file_path = "fault_detection_model.trt"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    builder.max_workspace_size = 1 << 20  # 1MB workspace size
    engine = builder.build_cuda_engine(network)

    with open(trt_file_path, 'wb') as f:
        f.write(engine.serialize())
    return trt_file_path

def infer_with_tensorrt(trt_file_path):
    with open(trt_file_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    input_shape = (1, 2)  # Adjust based on your input shape
    output_shape = (1, 1)  # Adjust based on your output shape

    d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().itemsize)
    d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().itemsize)
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # Prepare input data for inference
    input_data = np.random.rand(*input_shape).astype(np.float32)  # Example input data
    cuda.memcpy_htod_async(d_input, input_data, stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)

    stream.synchronize()
    print("Predictions: ", output_data)

if __name__ == "__main__":
    onnx_file_path = "fault_detection_model.onnx"
    trt_file_path = convert_to_tensorrt(onnx_file_path)
    infer_with_tensorrt(trt_file_path)
