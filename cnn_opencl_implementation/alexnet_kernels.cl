// 1) 2D Convolution
__kernel void conv2d(__global const float* input,
                     __global const float* weights,
                     __global const float* bias,
                     __global float* output,
                     const int width,
                     const int height,
                     const int kernel_size,
                     const int input_channels,
                     const int output_channels,
                     const int stride,
                     const int padding)
{
    int oc = get_global_id(0); // output channel
    int oy = get_global_id(1); // output y
    int ox = get_global_id(2); // output x

    int out_width = (width + 2*padding - kernel_size) / stride + 1;
    int out_height = (height + 2*padding - kernel_size) / stride + 1;

    if (oc >= output_channels || oy >= out_height || ox >= out_width) return;

    float sum = 0.0f;

    for (int ic = 0; ic < input_channels; ic++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = oy * stride + ky - padding;
                int in_x = ox * stride + kx - padding;

                float in_val = 0.0f;
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int in_idx = ic * width * height + in_y * width + in_x;
                    in_val = input[in_idx];
                }

                int w_idx = oc * input_channels * kernel_size * kernel_size
                            + ic * kernel_size * kernel_size
                            + ky * kernel_size + kx;
                float w_val = weights[w_idx];

                sum += in_val * w_val;
            }
        }
    }

    sum += bias[oc];

    int out_idx = oc * out_width * out_height + oy * out_width + ox;
    output[out_idx] = sum;
}

// 2) ReLU activation
__kernel void relu(__global const float* input,
                   __global float* output,
                   const int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

    float val = input[idx];
    output[idx] = val > 0.0f ? val : 0.0f;
}

// 3)Maxpooling
__kernel void maxpool(__global const float* input,
                      __global float* output,
                      const int channels,
                      const int input_height,
                      const int input_width,
                      const int kernel_size,
                      const int stride)

{
    int c = get_global_id(0);
    int oy = get_global_id(1);
    int ox = get_global_id(2);

    int out_width = (input_width - kernel_size) / stride + 1;
    int out_height = (input_height - kernel_size) / stride + 1;

    if (c >= channels || oy >= out_height || ox >= out_width) return;

    float max_val = -FLT_MAX;

    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            int in_y = oy * stride + ky;
            int in_x = ox * stride + kx;

            int in_idx = c * input_width * input_height + in_y * input_width + in_x;
            float val = input[in_idx];
            if (val > max_val) max_val = val;
        }
    }

    int out_idx = c * out_width * out_height + oy * out_width + ox;
    output[out_idx] = max_val;
}


__kernel void lrn(__global float* input_output,
                  const int width,
                  const int height,
                  const int channels,
                  const int local_size,
                  const float alpha,
                  const float beta,
                  const float k)
{
    int c = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);

    if (c >= channels || y >= height || x >= width) return;

    int half_size = local_size / 2;
    float sum = 0.0f;

    int idx = c * width * height + y * width + x;

    for (int i = c - half_size; i <= c + half_size; i++) {
        if (i >= 0 && i < channels) {
            int n_idx = i * width * height + y * width + x;
            float val = input_output[n_idx];
            sum += val * val;
        }
    }

    float scale = pow(k + alpha * sum, beta);
    input_output[idx] = input_output[idx] / scale;
}

// 5) Fully Connected Layer
__kernel void fully_connected(__global const float* input,
                              __global const float* weights,
                              __global const float* bias,
                              __global float* output,
                              const int input_size,
                              const int output_size)
{
    int oid = get_global_id(0);
    if (oid >= output_size) return;

    float sum = 0.0f;

    for (int i = 0; i < input_size; i++) {
        sum += input[i] * weights[oid * input_size + i];
    }

    sum += bias[oid];
    output[oid] = sum;
}


__kernel void softmax(__global float* input,__global float* output, const int size)
{
    int idx = get_global_id(0);
    if (idx >= size) return;

}


__kernel void fc(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    int in_features,
    int out_features)
{
    int gid = get_global_id(0);
    if (gid >= out_features)
        return;

    float result = bias[gid];
    for (int i = 0; i < in_features; i++) {
        result += input[i] * weight[gid * in_features + i];
    }
    output[gid] = result;
}
