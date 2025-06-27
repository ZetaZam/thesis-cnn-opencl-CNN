import pyopencl as cl
import numpy as np
from PIL import Image
import json
import time
import sys
import os

# --- Device selection based on your clinfo fixed IDs ---
def select_device_by_id(device_choice):
    platform_device_map = {
        'amd_igpu': (0, 0),
        'nvidia_gpu': (1, 0),
        'cpu': (2, 0),
    }

    if device_choice not in platform_device_map:
        raise ValueError(f"Invalid device_choice '{device_choice}'. Choose from {list(platform_device_map.keys())}")

    platform_id, device_id = platform_device_map[device_choice]

    platforms = cl.get_platforms()
    if platform_id >= len(platforms):
        raise RuntimeError(f"Platform ID {platform_id} out of range. {len(platforms)} platforms available.")

    platform = platforms[platform_id]
    devices = platform.get_devices()
    if device_id >= len(devices):
        raise RuntimeError(f"Device ID {device_id} out of range for platform {platform.name}.")

    return devices[device_id]


# --- Image preprocessing ---
def preprocess_image(image_path, size=(227, 227)):
    image = Image.open(image_path).convert("RGB").resize(size)
    image_np = np.array(image).astype(np.float32) / 255.0
    # Normalize per channel (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std
    # Change to CHW format
    image_np = image_np.transpose((2, 0, 1))
    return image_np.copy()  # ensure contiguous


# --- Load weights from npz file ---
def load_weights_npz(npz_path):
    weights = np.load(npz_path)
    print(f"Loaded weights from {npz_path}: keys = {list(weights.keys())}")
    return weights


# --- Load layer config from JSON ---
def load_layer_config(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    print(f"Loaded layer config with {len(config['layers'])} layers from {json_path}")
    return config


# --- Build OpenCL program from kernel file ---
def build_program(context, kernel_path):
    with open(kernel_path, "r") as f:
        kernel_source = f.read()
    program = cl.Program(context, kernel_source).build()
    print(f"Built OpenCL program from {kernel_path}")
    return program


# --- CNN layer runner ---
def run_cnn_layers(ctx, queue, program, config, weights, input_data):
    input_np = input_data.astype(np.float32)
    input_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_np)
    current_buf = input_buf
    current_shape = input_data.shape

    for i, layer in enumerate(config["layers"]):
        layer_type = layer["type"]
        print(f"\nExecuting layer {i}: {layer_type}")

        if layer_type == "conv":
            weight = weights[layer["weight"]].astype(np.float32)
            bias = weights[layer["bias"]].astype(np.float32)
            # Prepare buffers
            weight_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=weight)
            bias_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bias)
            out_h = layer["out_h"]
            out_w = layer["out_w"]
            out_c = layer["out_c"]
            output_np = np.empty((out_c, out_h, out_w), dtype=np.float32)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_np.nbytes)

            # Kernel execution
            kernel = getattr(program, "conv2d")
            kernel.set_args(
                current_buf, weight_buf, bias_buf, output_buf,
                np.int32(current_shape[2]),    # width
                np.int32(current_shape[1]),    # height
                np.int32(layer["kernel"]),     # kernel_size
                np.int32(current_shape[0]),    # input_channels
                np.int32(out_c),               # output_channels
                np.int32(layer["stride"]),     # stride
                np.int32(layer["padding"])     # padding
            )

            global_size = (out_c, out_h, out_w)
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            current_buf = output_buf
            current_shape = output_np.shape

        elif layer_type == "relu":
            output_np = np.empty(current_shape, dtype=np.float32)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_np.nbytes)

            kernel = getattr(program, "relu")
            kernel.set_args(current_buf, output_buf, np.int32(np.prod(current_shape)))

            global_size = (np.prod(current_shape),)
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            current_buf = output_buf

        elif layer_type == "maxpool" or layer_type == "pool":
            k = layer["kernel"]
            s = layer["stride"]
            c, h, w = current_shape
            out_h = (h - k) // s + 1
            out_w = (w - k) // s + 1
            output_np = np.empty((c, out_h, out_w), dtype=np.float32)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_np.nbytes)

            kernel = getattr(program, "maxpool")
            kernel.set_args(current_buf, output_buf,
                            np.int32(c), np.int32(h), np.int32(w),
                            np.int32(k), np.int32(s))

            global_size = (c, out_h, out_w)
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            current_buf = output_buf
            current_shape = output_np.shape

        elif layer_type == "fc":
            weight = weights[layer["weight"]].astype(np.float32)
            bias = weights[layer["bias"]].astype(np.float32)
            in_features = weight.shape[1]
            out_features = weight.shape[0]
            output_np = np.empty((out_features,), dtype=np.float32)

            # Copy current device buffer to host
            host_input = np.empty(current_shape, dtype=np.float32)
            cl.enqueue_copy(queue, host_input, current_buf)
            queue.finish()

            # Flatten explicitly in host
            host_input_flat = host_input.reshape(-1).astype(np.float32)

            # Create input buffer for FC kernel with flattened data
            input_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_input_flat)


            # Use current_buf directly as input buffer to kernel
            weight_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=weight)
            bias_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=bias)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_np.nbytes)

            kernel = getattr(program, "fc")
            kernel.set_args(current_buf, weight_buf, bias_buf, output_buf,
                            np.int32(in_features), np.int32(out_features))

            global_size = (out_features,)
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            current_buf = output_buf
            current_shape = output_np.shape

            

        elif layer_type == "softmax":
            output_np = np.empty(current_shape, dtype=np.float32)
            output_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, output_np.nbytes)

            kernel = getattr(program, "softmax")
            kernel.set_args(current_buf, output_buf, np.int32(current_shape[0]))

            global_size = (current_shape[0],)
            cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
            current_buf = output_buf

        elif layer_type == "flatten":
            # Flatten the current tensor to 1D
            c, h, w = current_shape
            flat_size = c * h * w

            # No kernel needed, just reshape metadata
            current_shape = (flat_size,)

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

    # Final result fetch
    final_output = np.empty(current_shape, dtype=np.float32)
    cl.enqueue_copy(queue, final_output, current_buf)
    return final_output

#def load_test_data(images_npy_path, labels_npy_path):
    test_images = np.load(images_npy_path)
    test_labels = np.load(labels_npy_path)

    # Normalize and transpose all images
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    processed = (test_images / 255.0 - mean[None, :, None, None]) / std[None, :, None, None]
    return processed.astype(np.float32), test_labels.astype(np.int64)

def load_test_data(images_dir, labels_path):
    # Load all .npy files from directory
    image_files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(".npy")
    ])

    test_images = np.array([np.load(f) for f in image_files])
    test_labels = np.load(labels_path)
    
    return test_images, test_labels

def main():
    # === Configurable parameters ===
    device_choice = 'nvidia_gpu'     # 'amd_igpu', 'nvidia_gpu', 'cpu'
    image_path = "lena_color_256.tif"
    weights_path = "alexnet_weights_pretrained.npz"
    config_path = "layer_config.json"
    kernel_path = "alexnet_kernels.cl"  # your OpenCL kernel source

    # --- Device selection ---
    try:
        device = select_device_by_id(device_choice)
    except Exception as e:
        print(f"Error selecting device: {e}")
        sys.exit(1)

    print(f"Using OpenCL device: {device.name} on platform {device.platform.name}")

    # --- Create context and command queue ---
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    # --- Load and preprocess image ---
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        sys.exit(1)

    input_data = preprocess_image(image_path)
    print(f"Preprocessed image shape: {input_data.shape}")

    # --- Load weights ---
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        sys.exit(1)

    weights = load_weights_npz(weights_path)

    # --- Load layer config ---
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_layer_config(config_path)

    # --- Build OpenCL program ---
    if not os.path.exists(kernel_path):
        print(f"Kernel source file not found: {kernel_path}")
        sys.exit(1)

    program = build_program(ctx, kernel_path)

    # --- Run CNN layers and benchmark ---
    # --- Device info (used later) ---
    try:
        compute_units = device.max_compute_units
    except AttributeError:
        compute_units = 'Unknown'

    # --- Benchmark inference on batch of test images ---
    images_npy_path = "/home/vaibhav/cnn_opencl_project/cifar10_resized/images/"
    labels_npy_path = "/home/vaibhav/cnn_opencl_project/data/labels.npy"

    if not (os.path.exists(images_npy_path) and os.path.exists(labels_npy_path)):
        print("Benchmark skipped: test image/label files not found.")
        return

    test_images, test_labels = load_test_data(images_npy_path, labels_npy_path)
    total_time = 0.0
    correct = 0
    num_samples = len(test_images)

    for i in range(num_samples):
        input_data = test_images[i]  # shape (3, 227, 227)
        label = test_labels[i]

        start = time.time()
        output = run_cnn_layers(ctx, queue, program, config, weights, input_data)
        end = time.time()
        total_time += (end - start)

        pred = int(np.argmax(output))
        if pred == label:
            correct += 1

        if i < 5:
            print(f"Sample {i+1}: Predicted={pred}, Actual={label}, Output={output[:5]}...")

    avg_time = total_time / num_samples
    accuracy = 100.0 * correct / num_samples

    print("\n========= Inference Benchmark Summary =========")
    print(f"Device: {device.name}")
    print(f"Compute Units: {compute_units}")
    print(f"Total Samples: {num_samples}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Time: {avg_time:.6f} seconds")



if __name__ == "__main__":
    main()
