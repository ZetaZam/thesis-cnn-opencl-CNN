# thesis-cnn-opencl-CNN

This repository contains two main components:

1. **OpenCL CNN Implementation**  
A modular OpenCL-based convolutional neural network (CNN) designed for inference using a heterogeneous computing pipeline. It supports execution on AMD CPUs, AMD iGPUs, and NVIDIA GPUs through OpenCL.

2. **Benchmarking Pipeline**  
A complete pipeline to benchmark CNN performance across different hardware backends.

since it uses cJSON, you need cJSON files to be in your project directory

4. **native implementation and dataset prep files**  
Includes native PyTorch-based inference for comparison, dataset preparation scripts, and profiling tools.




## Requirements

- Ubuntu 22.04 LTS  
- Python 3.x  
- PyOpenCL  
- PyTorch  
- OpenCL drivers (ROCm, CUDA, POCL)  
- Visual Studio Code (optional, used for development)
