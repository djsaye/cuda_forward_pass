/********************************************************************************************************
* IMPORTANT DIRECTIONS ON HOW TO USE :
*   Each define corresponds to a single optimization
*   To run an optimization, uncomment a define, and ONLY that define
*/

// #define BASELINE
// #define FP_16 // 4 points
// #define STREAMS // 4 points
// #define TILED // 2 points
// #define TUNING // 3 points
// #define MULT_LAYER // 1 point
// #define CONSTANT // 1 point


/* BELOW ARE OPTIMIZATIONS THAT ARE STACKED. same rules as above apply  */

// #define LAYER_TUNING
// #define TILED_TUNING
// #define TILED_LAYER_TUNING
// #define ATOMIC_TILED_LAYER_TUNING // FINAL SUBMISSION
#define TENSOR_CONST_TILED_LAYER_TUNING


/*****************************************************************************************************/

#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#ifdef FP_16
    #include "cuda_fp16.h"
#endif

#define TILE_SIZE 16

#ifdef STREAMS
    #define NUM_STREAMS 10
    static cudaStream_t stream_arr[NUM_STREAMS];
#endif

#ifdef TUNING
    #define K_LOOP 7
#endif

#ifdef TILED_TUNING
    #define K_LOOP 7
#endif

#ifdef MULT_LAYER
    #define C_LARGE 4
#endif

#ifdef LAYER_TUNING
    #define K_LOOP 7
    #define C_LARGE 4
#endif

#ifdef TILED_LAYER_TUNING
    #define K_LOOP 7
    #define C_LARGE 4
    #define SHARED_OUT 22
#endif

#ifdef ATOMIC_TILED_LAYER_TUNING
    #define K_LOOP 7
    #define C_LARGE 4
    #define SHARED_OUT 22
__constant__ float mask_constant[4096];
__shared__ float insh[SHARED_OUT * SHARED_OUT];
__shared__ float masksh[K_LOOP * K_LOOP];
#endif

#ifdef TENSOR_CONST_TILED_LAYER_TUNING
    #include "cuda_fp16.h"
    #define K_LOOP 7
    #define C_LARGE 4
    #define SHARED_OUT 22
__constant__ float mask_constant[4096];
__shared__ float insh[SHARED_OUT * SHARED_OUT];
#endif

#ifdef CONSTANT
__constant__ float mask_constant[4000];
#endif

#if defined(TUNING) || defined(TILED_TUNING) || defined(LAYER_TUNING) || defined(TILED_LAYER_TUNING) || defined(ATOMIC_TILED_LAYER_TUNING) || defined(TENSOR_CONST_TILED_LAYER_TUNING)
__global__ void conv_forward_kernel(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
#else
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
#endif
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
#if defined(BASELINE) || defined(STREAMS) 
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {

        float sum = 0.0;
        for (int c=0; c<Channel; c++) {
            for (int p=0; p<K; p++) {
                for (int q=0; q<K; q++) {
                    sum += in_4d(b,c,h+p,w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = sum;

    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(MULT_LAYER) 
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C_LARGE * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C_LARGE * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {

        float sum = 0.0;
#pragma unroll
        for (int c=0; c<C_LARGE; c++) {
            for (int p=0; p<K; p++) {
                for (int q=0; q<K; q++) {
                    sum += in_4d(b,c,h+p,w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = sum;

    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(TUNING) 
    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K_LOOP * K_LOOP) + (i2) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {

        float sum = 0.0;
        for (int c=0; c<Channel; c++) {
#pragma unroll
            for (int p=0; p<K_LOOP; p++) {
#pragma unroll
                for (int q=0; q<K_LOOP; q++) {
                    sum += in_4d(b,c,h+p,w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = sum;

    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(FP_16)
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {

        __half sum = __float2half(0.0);
        for (int c=0; c<Channel; c++) {
            for (int p=0; p<K; p++) {
                for (int q=0; q<K; q++) {
                    sum = __hfma(__float2half(in_4d(b,c,h+p,w+q)), __float2half(mask_4d(m,c,p,q)), sum);
                }
            }
        }
        out_4d(b,m,h,w) = __half2float(sum);

    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(TILED) 

    extern __shared__ float shmem[];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int shared_out = TILE_SIZE + K - 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);

    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int hbase = (blockIdx.y / W_size) * TILE_SIZE; // vertical base out data index for the block
    const int wbase = (blockIdx.y % W_size) * TILE_SIZE; // horizontal base out data index for the block

    const int m = blockIdx.x;
    const int h = hbase + h0;
    const int w = wbase + w0;
    const int b = blockIdx.z;

    float * in_shared = &shmem[0];
    float * mask_shared = &shmem[shared_out * shared_out];

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define in_2d(i1, i0) in_shared[(i1) * (shared_out) + i0]
    #define mask_2d(i1, i0) mask_shared[(i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    float sum = 0.0;
    for (int c=0; c<Channel; c++) {

        if (h0 < K && w0 < K && m < Map_out) {
            mask_2d(h0, w0) = mask_4d(m, c, h0, w0); 
        }

        if (b < Batch) {
            for (int i = h; i < hbase + shared_out; i += TILE_SIZE) {
                for (int j = w; j < wbase + shared_out; j += TILE_SIZE) {
                    in_2d(i - hbase, j - wbase) = in_4d(b, c, i, j);
                }
            }
        }
        __syncthreads();
            
        if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
            for (int p=0; p<K; p++) {
                for (int q=0; q<K; q++) {
                    sum += in_2d(h0+p, w0+q) * mask_2d(p, q);
                }
            }
        }
        __syncthreads();
    }

    if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
        out_4d(b,m,h,w) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(CONSTANT) 
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask_constant[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {

        float sum = 0.0;
        for (int c=0; c<Channel; c++) {
            for (int p=0; p<K; p++) {
                for (int q=0; q<K; q++) {
                    sum += in_4d(b,c,h+p,w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = sum;

    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(TILED_TUNING) 

    extern __shared__ float shmem[];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int shared_out = TILE_SIZE + K - 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);

    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int hbase = (blockIdx.y / W_size) * TILE_SIZE; // vertical base out data index for the block
    const int wbase = (blockIdx.y % W_size) * TILE_SIZE; // horizontal base out data index for the block

    const int m = blockIdx.x;
    const int h = hbase + h0;
    const int w = wbase + w0;
    const int b = blockIdx.z;

    float * in_shared = &shmem[0];
    float * mask_shared = &shmem[shared_out * shared_out];

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define in_2d(i1, i0) in_shared[(i1) * (shared_out) + i0]
    #define mask_2d(i1, i0) mask_shared[(i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    float sum = 0.0;
    for (int c=0; c<Channel; c++) {

        if (h0 < K && w0 < K && m < Map_out) {
            mask_2d(h0, w0) = mask_4d(m, c, h0, w0); 
        }

        if (b < Batch) {
            for (int i = h; i < hbase + shared_out; i += TILE_SIZE) {
                for (int j = w; j < wbase + shared_out; j += TILE_SIZE) {
                    in_2d(i - hbase, j - wbase) = in_4d(b, c, i, j);
                }
            }
        }
        __syncthreads();
            
        if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
#pragma unroll
            for (int p=0; p<K_LOOP; p++) {
#pragma unroll
                for (int q=0; q<K_LOOP; q++) {
                    sum += in_2d(h0+p, w0+q) * mask_2d(p, q);
                }
            }
        }
        __syncthreads();
    }

    if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
        out_4d(b,m,h,w) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(LAYER_TUNING) 
    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C_LARGE * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C_LARGE * K_LOOP * K_LOOP) + (i2) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {

        float sum = 0.0;
#pragma unroll
        for (int c=0; c<C_LARGE; c++) {
#pragma unroll
            for (int p=0; p<K_LOOP; p++) {
#pragma unroll
                for (int q=0; q<K_LOOP; q++) {
                    sum += in_4d(b,c,h+p,w+q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = sum;

    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(TILED_LAYER_TUNING) 

    extern __shared__ float shmem[];

    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);

    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int hbase = (blockIdx.y / W_size) * TILE_SIZE; // vertical base out data index for the block
    const int wbase = (blockIdx.y % W_size) * TILE_SIZE; // horizontal base out data index for the block

    const int m = blockIdx.x;
    const int h = hbase + h0;
    const int w = wbase + w0;
    const int b = blockIdx.z;
    // const int check = (int) (h < Height_out && w < Width_out);

    float * __restrict__ in_shared = &shmem[0];
    float * __restrict__ mask_shared = &shmem[SHARED_OUT * SHARED_OUT];

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C_LARGE * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C_LARGE * K_LOOP * K_LOOP) + (i2) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]
    #define in_2d(i1, i0) in_shared[(i1) * (SHARED_OUT) + i0]
    #define mask_2d(i1, i0) mask_shared[(i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    float sum = 0.0;
#pragma unroll
    for (int c=0; c<C_LARGE; c++) {

        if (h0 < K_LOOP && w0 < K_LOOP && m < Map_out) {
            mask_2d(h0, w0) = mask_4d(m, c, h0, w0); 
        }

        if (b < Batch) {
            for (int i = h; i < hbase + SHARED_OUT; i += TILE_SIZE) {
                for (int j = w; j < wbase + SHARED_OUT; j += TILE_SIZE) {
                    in_2d(i - hbase, j - wbase) = in_4d(b, c, i, j);
                }
            }
        }
        __syncthreads();
            
        if (h < Height_out && w < Width_out && b <  Batch && m < Map_out) {
#pragma unroll
            for (int p=0; p<K_LOOP; p++) {
#pragma unroll
                for (int q=0; q<K_LOOP; q++) {
                    sum += in_2d(h0+p, w0+q) * mask_2d(p, q);
                }
            }
        }
        __syncthreads();
    }

    if (h < Height_out && w < Width_out && b < Batch && m < Map_out) {
        out_4d(b,m,h,w) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
#endif

#if defined(ATOMIC_TILED_LAYER_TUNING) 

    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    // const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);

    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int hbase = (blockIdx.y / W_size) * TILE_SIZE; // vertical base out data index for the block
    const int wbase = (blockIdx.y % W_size) * TILE_SIZE; // horizontal base out data index for the block

    const int m = blockIdx.x;
    const int h = hbase + h0;
    const int w = wbase + w0;
    const int b = blockIdx.z;

    float * __restrict__ in_shared = insh;
    float * __restrict__ mask_shared = masksh;
    const float * __restrict__ maskcons = mask_constant;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C_LARGE * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) maskcons[(i3) * (C_LARGE * K_LOOP * K_LOOP) + (i2) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]
    #define in_2d(i1, i0) in_shared[(i1) * (SHARED_OUT) + i0]
    #define mask_2d(i1, i0) mask_shared[(i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    float sum = 0.0;

#pragma unroll
    for(int c=0; c < C_LARGE; c++) {
        // if (h0 < K_LOOP && w0 < K_LOOP) {
        //     mask_2d(h0, w0) = mask_4d(m, c, h0, w0); 
        // }

        for (int i = h; i < hbase + SHARED_OUT; i += TILE_SIZE) {
            for (int j = w; j < wbase + SHARED_OUT; j += TILE_SIZE) {
                in_2d(i - hbase, j - wbase) = in_4d(b, c, i, j);
            }
        }
        __syncthreads();
            
        if (h < Height_out && w < Width_out) {
#pragma unroll
            for (int p=0; p<K_LOOP; p++) {
#pragma unroll
                for (int q=0; q<K_LOOP; q++) {
                    sum += in_2d(h0+p, w0+q) * mask_4d(m, c, p, q);
                }
            }

        }
    }

    if (h < Height_out && w < Width_out) {
        out_4d(b,m,h,w) = sum;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef mask_2d
    #undef in_2d
#endif
}




#if defined(MULT_LAYER) 
__global__ void conv_forward_kernel_small(float * output, const float * input, const float * mask, const int Batch, const int Map_out, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {
        float sum = 0.0;
        for (int p=0; p<K; p++) {
            for (int q=0; q<K; q++) {
                sum += in_4d(b,0,h+p,w+q) * mask_4d(m,0,p,q);
            }
        }
        out_4d(b,m,h,w) = sum;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
#endif

#if defined(LAYER_TUNING) 
__global__ void conv_forward_kernel_small(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int Batch, const int Map_out, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);


    const int m = blockIdx.x;
    const int h = (blockIdx.y / W_size) * TILE_SIZE + threadIdx.y;
    const int w = (blockIdx.y % W_size) * TILE_SIZE + threadIdx.x;
    const int b = blockIdx.z;


    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    if (m < Map_out && h < Height_out && w < Width_out && b < Batch) {
        float sum = 0.0;
#pragma unroll
        for (int p=0; p<K_LOOP; p++) {
#pragma unroll
            for (int q=0; q<K_LOOP; q++) {
                sum += in_4d(b,0,h+p,w+q) * mask_4d(m,0,p,q);
            }
        }
        out_4d(b,m,h,w) = sum;
    }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
#endif

#if defined(TILED_LAYER_TUNING) 
__global__ void conv_forward_kernel_small(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int Batch, const int Map_out, const int Height, const int Width)
{
    extern __shared__ float shmem[];

    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);

    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int hbase = (blockIdx.y / W_size) * TILE_SIZE; // vertical base out data index for the block
    const int wbase = (blockIdx.y % W_size) * TILE_SIZE; // horizontal base out data index for the block

    const int m = blockIdx.x;
    const int h = hbase + h0;
    const int w = wbase + w0;
    const int b = blockIdx.z;
    // const int check = (int) (h < Height_out && w < Width_out);

    float * __restrict__ in_shared = &shmem[0];
    float * __restrict__ mask_shared = &shmem[SHARED_OUT * SHARED_OUT];

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Height * Width) + (i1) * (Width) + i0]
#if defined(TILED_LAYER_TUNING)
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]
#else
    #define mask_4d(i3, i2, i1, i0) mask_constant[(i3) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]
#endif
    #define in_2d(i1, i0) in_shared[(i1) * (SHARED_OUT) + i0]
    #define mask_2d(i1, i0) mask_shared[(i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    float sum = 0.0;
    if (h0 < K_LOOP && w0 < K_LOOP) {
        mask_2d(h0, w0) = mask_4d(m, 0, h0, w0); 
    }

    for (int i = h; i < hbase + SHARED_OUT; i += TILE_SIZE) {
        for (int j = w; j < wbase + SHARED_OUT; j += TILE_SIZE) {
            in_2d(i - hbase, j - wbase) = in_4d(b, 0, i, j);
        }
    }
    __syncthreads();
        
    if (h < Height_out && w < Width_out) {
#pragma unroll
        for (int p=0; p<K_LOOP; p++) {
#pragma unroll
            for (int q=0; q<K_LOOP; q++) {
                sum += in_2d(h0+p, w0+q) * mask_2d(p, q);
            }
        }
        out_4d(b,m,h,w) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef mask_2d
    #undef in_2d
 
}
#endif

#if defined(ATOMIC_TILED_LAYER_TUNING) || defined(CONST_ATOMIC_TILED_LAYER_TUNING) 
__global__ void conv_forward_kernel_small(float * __restrict__ output, const float * __restrict__ input, const float * __restrict__ mask, const int Batch, const int Map_out, const int Height, const int Width)
{
    // extern __shared__ float shmem[];

    const int Height_out = Height - K_LOOP + 1;
    const int Width_out = Width - K_LOOP + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    const int W_size = ceil((float) (Width_out) / (float) TILE_SIZE);
    // const int H_size = ceil((float) (Height_out) / (float) TILE_SIZE);

    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;
    const int hbase = (blockIdx.y / W_size) * TILE_SIZE; // vertical base out data index for the block
    const int wbase = (blockIdx.y % W_size) * TILE_SIZE; // horizontal base out data index for the block

    const int m = blockIdx.x;
    const int h = hbase + h0;
    const int w = wbase + w0;
    const int b = blockIdx.z;
    // const int check = (int) (h < Height_out && w < Width_out);

    float * __restrict__ in_shared = insh;
    // float * __restrict__ mask_shared = masksh;
    const float * __restrict__ maskcons = mask_constant;
    // __shared__ float mask_shared[K_LOOP * K_LOOP];

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) maskcons[(i3) * (K_LOOP * K_LOOP) + (i1) * (K_LOOP) + i0]
    #define in_2d(i1, i0) in_shared[(i1) * (SHARED_OUT) + i0]
    #define mask_2d(i1, i0) mask_shared[(i1) * (K_LOOP) + i0]

    // Insert your GPU convolution kernel code here

    float sum = 0.0;
    // if (h0 < K_LOOP && w0 < K_LOOP) {
    //     mask_2d(h0, w0) = mask_4d(m, 0, h0, w0); 
    // }

    for (int i = h; i < hbase + SHARED_OUT; i += TILE_SIZE) {
        for (int j = w; j < wbase + SHARED_OUT; j += TILE_SIZE) {
            in_2d(i - hbase, j - wbase) = in_4d(b, 0, i, j);
        }
    }
    __syncthreads();
        
    if (h < Height_out && w < Width_out) {
#pragma unroll
        for (int p=0; p<K_LOOP; p++) {
#pragma unroll
            for (int q=0; q<K_LOOP; q++) {
                sum += in_2d(h0+p, w0+q) * mask_4d(m, 0, p, q);
            }
        }
        out_4d(b,m,h,w) = sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef mask_2d
    #undef in_2d
 
}
#endif


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
#if defined(BASELINE) || defined(FP_16) || defined(TILED) || defined(TUNING) || defined(TILED_TUNING) || defined(MULT_LAYER) || defined(LAYER_TUNING)  || defined(TILED_LAYER_TUNING)

    const unsigned int out_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    const unsigned int in_size = Batch * Channel * Height * Width;
    const unsigned int mask_size = Map_out * Channel * K * K;
 

    cudaMalloc((void **) device_input_ptr,  in_size * sizeof(float));
    cudaMalloc((void **) device_output_ptr, out_size * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, mask_size * sizeof(float));

    cudaMemcpy((void *) *device_input_ptr, (void *) host_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) *device_mask_ptr, (void *) host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
#endif

#if defined(CONSTANT)

    const unsigned int out_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    const unsigned int in_size = Batch * Channel * Height * Width;
    const unsigned int mask_size = Map_out * Channel * K * K;
 

    cudaMalloc((void **) device_input_ptr,  in_size * sizeof(float));
    cudaMalloc((void **) device_output_ptr, out_size * sizeof(float));

    cudaMemcpy((void *) *device_input_ptr, (void *) host_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_constant, host_mask, mask_size * sizeof(float));
#endif

#if defined(STREAMS)
    const unsigned int out_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    const unsigned int in_size = Batch * Channel * Height * Width;
    const unsigned int mask_size = Map_out * Channel * K * K;

    const unsigned int in_section = ceil(((float) in_size) / ((float) NUM_STREAMS));
    const int batch_section = ceil(((float) Batch) / ((float) NUM_STREAMS));
    
    cudaMalloc((void **) device_input_ptr,  in_size * sizeof(float));
    cudaMalloc((void **) device_output_ptr, out_size * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, mask_size * sizeof(float));

    cudaMemcpy((void *) *device_mask_ptr, (void *) host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    for (int i=0; i<NUM_STREAMS; i++) {
        cudaStreamCreate(&stream_arr[i]);
        cudaMemcpyAsync((void *) (*device_input_ptr + i*in_section), (void *) (host_input + i*in_section), in_section * sizeof(float), cudaMemcpyHostToDevice, stream_arr[i]);
    }
#endif

#if defined(ATOMIC_TILED_LAYER_TUNING)

    const unsigned int out_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    const unsigned int in_size = Batch * Channel * Height * Width;
    const unsigned int mask_size = Map_out * Channel * K * K;
 

    cudaMalloc((void **) device_input_ptr,  in_size * sizeof(float));
    cudaMalloc((void **) device_output_ptr, out_size * sizeof(float));
    // cudaMalloc((void **) device_mask_ptr, mask_size * sizeof(float));

    cudaMemcpy((void *) *device_input_ptr, (void *) host_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy((void *) *device_mask_ptr, (void *) host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_constant, host_mask, mask_size * sizeof(float));
#endif
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
#if defined(BASELINE) || defined(FP_16) || defined(TUNING)
    // Set the kernel dimensions and call the kernel
    const int Height_size = ceil((float) (Height - K + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K + 1) / (float) TILE_SIZE);
    //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, Batch);

    conv_forward_kernel<<<gridSize, blockSize>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
#endif

#if defined(TILED) || defined(TILED_TUNING)
    // Set the kernel dimensions and call the kernel
    const int Height_size = ceil((float) (Height - K + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K + 1) / (float) TILE_SIZE);
    //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

    const int shmem_size = sizeof(float) * ( ((TILE_SIZE + K-1)*(TILE_SIZE + K-1)) + (K*K) );

    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, Batch);

    // printf("This is K: %d\n", K);

    conv_forward_kernel<<<gridSize, blockSize, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
#endif

#if defined(STREAMS)
    const int batch_section = ceil(((float) Batch) / ((float) NUM_STREAMS));
    // const unsigned int mask_section = ceil(((float) mask_size) / ((float) NUM_STREAMS));

    const unsigned int out_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    const unsigned int in_size = Batch * Channel * Height * Width;

    const unsigned int out_section = ceil(((float) out_size) / ((float) NUM_STREAMS));
    const unsigned int in_section = ceil(((float) in_size) / ((float) NUM_STREAMS));

    const int Height_size = ceil((float) (Height - K + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K + 1) / (float) TILE_SIZE);
    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, batch_section);

    for (int i=0; i<NUM_STREAMS; i++) {
        conv_forward_kernel<<<gridSize, blockSize, 0, stream_arr[i]>>>(device_output + i*out_section, device_input + i*in_section, device_mask, Batch, Map_out, Channel, Height, Width, K);
    }
#endif

#if defined(CONSTANT)
     // Set the kernel dimensions and call the kernel
    const int Height_size = ceil((float) (Height - K + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K + 1) / (float) TILE_SIZE);
    //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, Batch);

    conv_forward_kernel<<<gridSize, blockSize>>>(device_output, device_input, NULL, Batch, Map_out, Channel, Height, Width, K);
#endif

#if defined(MULT_LAYER) || defined(LAYER_TUNING)
    // Set the kernel dimensions and call the kernel
    const int Height_size = ceil((float) (Height - K + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K + 1) / (float) TILE_SIZE);
    //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, Batch);

    if (Channel == 4)
        conv_forward_kernel<<<gridSize, blockSize>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    else
        conv_forward_kernel_small<<<gridSize, blockSize>>>(device_output, device_input, device_mask, Batch, Map_out, Height, Width, K);
#endif

#if defined(TILED_LAYER_TUNING)
    // Set the kernel dimensions and call the kernel
    const int Height_size = ceil((float) (Height - K_LOOP + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K_LOOP + 1) / (float) TILE_SIZE);
    //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

    const int shmem_size = sizeof(float) * ( (SHARED_OUT*SHARED_OUT) + (K_LOOP*K_LOOP) );


    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, Batch);

    if (Channel == C_LARGE)
        conv_forward_kernel<<<gridSize, blockSize, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    else
        conv_forward_kernel_small<<<gridSize, blockSize, shmem_size>>>(device_output, device_input, device_mask, Batch, Map_out, Height, Width);
#endif

#if defined(ATOMIC_TILED_LAYER_TUNING)
    // Set the kernel dimensions and call the kernel
    const int Height_size = ceil((float) (Height - K_LOOP + 1) / (float) TILE_SIZE);
    const int Width_size = ceil((float) (Width - K_LOOP + 1) / (float) TILE_SIZE);
    //const int Batch_size = ceil((float) Batch / (float) TILE_SIZE);

    // const int shmem_size = sizeof(float) * Channel * ( (SHARED_OUT*SHARED_OUT) + (K_LOOP*K_LOOP) );


    const int Y = Height_size * Width_size;

    dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridSize(Map_out, Y, Batch);

    if (Channel == C_LARGE)
        conv_forward_kernel<<<gridSize, blockSize>>>(device_output, device_input, NULL, Batch, Map_out, Channel, Height, Width, K);
    else
        conv_forward_kernel_small<<<gridSize, blockSize>>>(device_output, device_input, NULL, Batch, Map_out, Height, Width);
#endif
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
#if defined(BASELINE) || defined(FP_16) || defined(TILED) || defined(TUNING) || defined(TILED_TUNING) || defined(MULT_LAYER) || defined(LAYER_TUNING) || defined(TILED_LAYER_TUNING)
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    
    cudaMemcpy((void *) host_output, (void *) device_output, Height_out * Width_out * Batch * Map_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree((void *) device_input);
    cudaFree((void *) device_mask);
    cudaFree((void *) device_output);
#endif

#if defined(STREAMS)
    const unsigned int out_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);
    const unsigned int out_section = ceil(((float) out_size) / ((float) NUM_STREAMS));

    
    for (int i=0; i<NUM_STREAMS; i++) {
        cudaMemcpyAsync((void *) (host_output + i*out_section), (void *) (device_output + i*out_section), out_section * sizeof(float), cudaMemcpyDeviceToHost, stream_arr[i]);
        cudaStreamDestroy(stream_arr[i]);
    }
    // Free device memory
    cudaFree((void *) device_input);
    cudaFree((void *) device_mask);
    cudaFree((void *) device_output);
#endif

#if defined(CONSTANT) || defined(ATOMIC_TILED_LAYER_TUNING)
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    
    cudaMemcpy((void *) host_output, (void *) device_output, Height_out * Width_out * Batch * Map_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree((void *) device_input);
    cudaFree((void *) device_output);
#endif
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
