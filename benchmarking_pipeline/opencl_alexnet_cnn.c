#define CL_TARGET_OPENCL_VERSION 120
#include "cJSON.h"
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// AlexNet Configuration
#define INPUT_WIDTH 227
#define INPUT_HEIGHT 227
#define INPUT_CHANNELS 3
#define NUM_CLASSES 1000

// OpenCL configuration
#define MAX_SOURCE_SIZE (0x100000)
#define WORK_GROUP_SIZE 256

// Network parameters
typedef struct {
    cl_mem weights;
    cl_mem biases;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
} ConvLayer;

typedef struct {
    int kernel_size;
    int stride;
} PoolLayer;

typedef struct {
    cl_mem weights;
    cl_mem biases;
    int in_features;
    int out_features;
} FCLayer;

// OpenCL objects
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel conv_kernel, pool_kernel, fc_kernel, relu_kernel, lrn_kernel, softmax_kernel;

// Buffers
cl_mem input_buffer, output_buffer;
cl_mem intermediate_buffers[20]; // For layer outputs

// Error checking
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

// Load OpenCL kernel source
char* load_kernel_source(const char* filename, size_t* size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Failed to load kernel source: %s\n", filename);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);

    char* source = (char*)malloc(*size + 1);
    fread(source, 1, *size, fp);
    source[*size] = '\0';
    fclose(fp);
    return source;
}

// Initialize OpenCL
void init_opencl() {
    cl_int err;

    // Get platform and device
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    // Load and build kernel
    size_t source_size;
    char* source_str = load_kernel_source("alexnet_kernels.cl", &source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, &source_size, &err);
    CHECK_ERROR(err);
    free(source_str);

    // Build with optimization
    const char options[] = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error:\n%s\n", log);
        free(log);
        exit(1);
    }

    // Create kernels
    conv_kernel = clCreateKernel(program, "conv2d", &err); CHECK_ERROR(err);
    pool_kernel = clCreateKernel(program, "maxpool", &err); CHECK_ERROR(err);
    fc_kernel = clCreateKernel(program, "fc", &err); CHECK_ERROR(err);
    relu_kernel = clCreateKernel(program, "relu", &err); CHECK_ERROR(err);
    lrn_kernel = clCreateKernel(program, "lrn", &err); CHECK_ERROR(err);
    softmax_kernel = clCreateKernel(program, "softmax", &err); CHECK_ERROR(err);
}

// Load weights from file (simplified)
void load_weights(const char* filename, ConvLayer* conv_layers, FCLayer* fc_layers) {
    // In a real implementation, you would load weights from file
    // Here we just initialize with zeros for demonstration
    
    for (int i = 0; i < 5; i++) {
        size_t weight_size = conv_layers[i].in_channels * conv_layers[i].out_channels * 
                            conv_layers[i].kernel_size * conv_layers[i].kernel_size * sizeof(float);
        size_t bias_size = conv_layers[i].out_channels * sizeof(float);
        
        conv_layers[i].weights = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, NULL);
        conv_layers[i].biases = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, NULL);
    }
    
    for (int i = 0; i < 3; i++) {
        size_t weight_size = fc_layers[i].in_features * fc_layers[i].out_features * sizeof(float);
        size_t bias_size = fc_layers[i].out_features * sizeof(float);
        
        fc_layers[i].weights = clCreateBuffer(context, CL_MEM_READ_ONLY, weight_size, NULL, NULL);
        fc_layers[i].biases = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size, NULL, NULL);
    }
}

// Initialize AlexNet layers
void init_alexnet(ConvLayer* conv_layers, PoolLayer* pool_layers, FCLayer* fc_layers) {
    // Conv1
    conv_layers[0] = (ConvLayer){NULL, NULL, 3, 64, 11, 4, 0};
    pool_layers[0] = (PoolLayer){3, 2};
    
    // Conv2
    conv_layers[1] = (ConvLayer){NULL, NULL, 64, 192, 5, 1, 2};
    pool_layers[1] = (PoolLayer){3, 2};
    
    // Conv3
    conv_layers[2] = (ConvLayer){NULL, NULL, 192, 384, 3, 1, 1};
    
    // Conv4
    conv_layers[3] = (ConvLayer){NULL, NULL, 384, 256, 3, 1, 1};
    
    // Conv5
    conv_layers[4] = (ConvLayer){NULL, NULL, 256, 256, 3, 1, 1};
    pool_layers[2] = (PoolLayer){3, 2};
    
    // FC layers
    fc_layers[0] = (FCLayer){NULL, NULL, 256*6*6, 4096};
    fc_layers[1] = (FCLayer){NULL, NULL, 4096, 4096};
    fc_layers[2] = (FCLayer){NULL, NULL, 4096, NUM_CLASSES};
    
    // Load weights (in real implementation, load from file)
    load_weights("alexnet_weights.bin", conv_layers, fc_layers);
}

// Run convolution layer
void run_conv_layer(cl_mem input, cl_mem output, ConvLayer* layer, int width, int height) {
    cl_int err;
    
    // Set kernel arguments
    err = clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &layer->weights); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 2, sizeof(cl_mem), &layer->biases); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 3, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 4, sizeof(int), &width); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 5, sizeof(int), &height); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 6, sizeof(int), &layer->kernel_size); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 7, sizeof(int), &layer->in_channels); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 8, sizeof(int), &layer->out_channels); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 9, sizeof(int), &layer->stride); CHECK_ERROR(err);
    err = clSetKernelArg(conv_kernel, 10, sizeof(int), &layer->padding); CHECK_ERROR(err);
    
    // Calculate output dimensions
    int out_width = (width + 2*layer->padding - layer->kernel_size) / layer->stride + 1;
    int out_height = (height + 2*layer->padding - layer->kernel_size) / layer->stride + 1;
    
    // Execute kernel
    size_t global_size[3] = {layer->out_channels, out_height, out_width};
    err = clEnqueueNDRangeKernel(queue, conv_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// Run pooling layer
void run_pool_layer(cl_mem input, cl_mem output, PoolLayer* layer, int channels, int width, int height) {
    cl_int err;
    
    // Set kernel arguments
    err = clSetKernelArg(pool_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(pool_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(pool_kernel, 2, sizeof(int), &channels); CHECK_ERROR(err);
    err = clSetKernelArg(pool_kernel, 3, sizeof(int), &width); CHECK_ERROR(err);
    err = clSetKernelArg(pool_kernel, 4, sizeof(int), &height); CHECK_ERROR(err);
    err = clSetKernelArg(pool_kernel, 5, sizeof(int), &layer->kernel_size); CHECK_ERROR(err);
    err = clSetKernelArg(pool_kernel, 6, sizeof(int), &layer->stride); CHECK_ERROR(err);
    
    // Calculate output dimensions
    int out_width = (width - layer->kernel_size) / layer->stride + 1;
    int out_height = (height - layer->kernel_size) / layer->stride + 1;
    
    // Execute kernel
    size_t global_size[3] = {channels, out_height, out_width};
    err = clEnqueueNDRangeKernel(queue, pool_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// Run fully connected layer
void run_fc_layer(cl_mem input, cl_mem output, FCLayer* layer) {
    cl_int err;
    
    // Set kernel arguments
    err = clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), &layer->weights); CHECK_ERROR(err);
    err = clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), &layer->biases); CHECK_ERROR(err);
    err = clSetKernelArg(fc_kernel, 3, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(fc_kernel, 4, sizeof(int), &layer->in_features); CHECK_ERROR(err);
    err = clSetKernelArg(fc_kernel, 5, sizeof(int), &layer->out_features); CHECK_ERROR(err);
    
    // Execute kernel
    size_t global_size = layer->out_features;
    err = clEnqueueNDRangeKernel(queue, fc_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// Run ReLU activation
void run_relu(cl_mem input, cl_mem output, int size) {
    cl_int err;
    
    err = clSetKernelArg(relu_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(relu_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(relu_kernel, 2, sizeof(int), &size); CHECK_ERROR(err);
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue, relu_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// Run LRN layer
void run_lrn(cl_mem input_output, int width, int height, int channels) {
    cl_int err;
    int local_size = 5;
    float alpha = 0.0001f, beta = 0.75f, k = 2.0f;
    
    err = clSetKernelArg(lrn_kernel, 0, sizeof(cl_mem), &input_output); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 1, sizeof(int), &width); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 2, sizeof(int), &height); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 3, sizeof(int), &channels); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 4, sizeof(int), &local_size); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 5, sizeof(float), &alpha); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 6, sizeof(float), &beta); CHECK_ERROR(err);
    err = clSetKernelArg(lrn_kernel, 7, sizeof(float), &k); CHECK_ERROR(err);
    
    size_t global_size[3] = {channels, height, width};
    err = clEnqueueNDRangeKernel(queue, lrn_kernel, 3, NULL, global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// Run softmax
void run_softmax(cl_mem input, cl_mem output, int size) {
    cl_int err;
    
    err = clSetKernelArg(softmax_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(softmax_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(softmax_kernel, 2, sizeof(int), &size); CHECK_ERROR(err);
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue, softmax_kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// Main AlexNet inference function
float* run_alexnet(float* input_image, ConvLayer* conv_layers, PoolLayer* pool_layers, FCLayer* fc_layers) {
    cl_int err;
    
    // Create input buffer
    input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                INPUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), 
                                input_image, &err);
    CHECK_ERROR(err);
    
    // Create intermediate buffers
    for (int i = 0; i < 20; i++) {
        intermediate_buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                              INPUT_WIDTH * INPUT_HEIGHT * 512 * sizeof(float), 
                                              NULL, &err);
        CHECK_ERROR(err);
    }
    
    // Create output buffer
    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                 NUM_CLASSES * sizeof(float), NULL, &err);
    CHECK_ERROR(err);
    
    // AlexNet architecture
    // Layer 1: Conv -> ReLU -> LRN -> Pool
    run_conv_layer(input_buffer, intermediate_buffers[0], &conv_layers[0], INPUT_WIDTH, INPUT_HEIGHT);
    run_relu(intermediate_buffers[0], intermediate_buffers[1], 64*55*55);
    run_lrn(intermediate_buffers[1], 55, 55, 64);
    run_pool_layer(intermediate_buffers[1], intermediate_buffers[2], &pool_layers[0], 64, 55, 55);
    
    // Layer 2: Conv -> ReLU -> LRN -> Pool
    run_conv_layer(intermediate_buffers[2], intermediate_buffers[3], &conv_layers[1], 27, 27);
    run_relu(intermediate_buffers[3], intermediate_buffers[4], 192*27*27);
    run_lrn(intermediate_buffers[4], 27, 27, 192);
    run_pool_layer(intermediate_buffers[4], intermediate_buffers[5], &pool_layers[1], 192, 27, 27);
    
    // Layer 3: Conv -> ReLU
    run_conv_layer(intermediate_buffers[5], intermediate_buffers[6], &conv_layers[2], 13, 13);
    run_relu(intermediate_buffers[6], intermediate_buffers[7], 384*13*13);
    
    // Layer 4: Conv -> ReLU
    run_conv_layer(intermediate_buffers[7], intermediate_buffers[8], &conv_layers[3], 13, 13);
    run_relu(intermediate_buffers[8], intermediate_buffers[9], 256*13*13);
    
    // Layer 5: Conv -> ReLU -> Pool
    run_conv_layer(intermediate_buffers[9], intermediate_buffers[10], &conv_layers[4], 13, 13);
    run_relu(intermediate_buffers[10], intermediate_buffers[11], 256*13*13);
    run_pool_layer(intermediate_buffers[11], intermediate_buffers[12], &pool_layers[2], 256, 13, 13);
    
    // Flatten (handled implicitly by using 1D buffers)
    
    // FC layers
    run_fc_layer(intermediate_buffers[12], intermediate_buffers[13], &fc_layers[0]);
    run_relu(intermediate_buffers[13], intermediate_buffers[14], 4096);
    
    run_fc_layer(intermediate_buffers[14], intermediate_buffers[15], &fc_layers[1]);
    run_relu(intermediate_buffers[15], intermediate_buffers[16], 4096);
    
    run_fc_layer(intermediate_buffers[16], intermediate_buffers[17], &fc_layers[2]);
    
    // Softmax
    run_softmax(intermediate_buffers[17], output_buffer, NUM_CLASSES);
    
    // Read back results
    float* results = (float*)malloc(NUM_CLASSES * sizeof(float));
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, 
                            NUM_CLASSES * sizeof(float), results, 0, NULL, NULL);
    CHECK_ERROR(err);
    
    return results;
}

// Cleanup
void cleanup(ConvLayer* conv_layers, FCLayer* fc_layers) {
    for (int i = 0; i < 5; i++) {
        clReleaseMemObject(conv_layers[i].weights);
        clReleaseMemObject(conv_layers[i].biases);
    }
    for (int i = 0; i < 3; i++) {
        clReleaseMemObject(fc_layers[i].weights);
        clReleaseMemObject(fc_layers[i].biases);
    }
    for (int i = 0; i < 20; i++) {
        clReleaseMemObject(intermediate_buffers[i]);
    }
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    
    clReleaseKernel(conv_kernel);
    clReleaseKernel(pool_kernel);
    clReleaseKernel(fc_kernel);
    clReleaseKernel(relu_kernel);
    clReleaseKernel(lrn_kernel);
    clReleaseKernel(softmax_kernel);
    
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


void load_config_from_json(const char* filename, ConvLayer* conv_layers, PoolLayer* pool_layers, FCLayer* fc_layers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open JSON file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* data = (char*)malloc(length + 1);
    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON* root = cJSON_Parse(data);
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", cJSON_GetErrorPtr());
        exit(EXIT_FAILURE);
    }

    cJSON* convs = cJSON_GetObjectItem(root, "conv_layers");
    for (int i = 0; i < cJSON_GetArraySize(convs); i++) {
        cJSON* layer = cJSON_GetArrayItem(convs, i);
        conv_layers[i].in_channels = cJSON_GetObjectItem(layer, "in_channels")->valueint;
        conv_layers[i].out_channels = cJSON_GetObjectItem(layer, "out_channels")->valueint;
        conv_layers[i].kernel_size = cJSON_GetObjectItem(layer, "kernel_size")->valueint;
        conv_layers[i].stride = cJSON_GetObjectItem(layer, "stride")->valueint;
        conv_layers[i].padding = cJSON_GetObjectItem(layer, "padding")->valueint;
    }

    cJSON* pools = cJSON_GetObjectItem(root, "pool_layers");
    for (int i = 0; i < cJSON_GetArraySize(pools); i++) {
        cJSON* layer = cJSON_GetArrayItem(pools, i);
        pool_layers[i].kernel_size = cJSON_GetObjectItem(layer, "kernel_size")->valueint;
        pool_layers[i].stride = cJSON_GetObjectItem(layer, "stride")->valueint;
    }

    cJSON* fcs = cJSON_GetObjectItem(root, "fc_layers");
    for (int i = 0; i < cJSON_GetArraySize(fcs); i++) {
        cJSON* layer = cJSON_GetArrayItem(fcs, i);
        fc_layers[i].in_features = cJSON_GetObjectItem(layer, "in_features")->valueint;
        fc_layers[i].out_features = cJSON_GetObjectItem(layer, "out_features")->valueint;
    }

    cJSON_Delete(root);
    free(data);
}
int main() {
    init_opencl();

    ConvLayer conv_layers[MAX_LAYERS];
    PoolLayer pool_layers[MAX_LAYERS];
    FCLayer fc_layers[MAX_LAYERS];

    load_config_from_json("alexnet_config.json", conv_layers, pool_layers, fc_layers);
    load_weights("alexnet_weights.bin", conv_layers, fc_layers);

    float* input_image = (float*)calloc(INPUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT, sizeof(float));
    float* results = run_alexnet(input_image, conv_layers, pool_layers, fc_layers);

    int max_idx = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (results[i] > results[max_idx]) max_idx = i;
    }
    printf("Predicted class: %d with confidence: %f\n", max_idx, results[max_idx]);

    free(input_image);
    free(results);
    cleanup(conv_layers, fc_layers);
    return 0;
}
