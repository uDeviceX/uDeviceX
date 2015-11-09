/*
 *  main.cu
 *  ctc falcon
 *
 *  Created by Dmitry Alexeev on Oct 28, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

//
//  main.cpp
//  cpudpd
//
//  Created by Dmitry Alexeev on 26/10/15.
//  Copyright © 2015 Dmitry Alexeev. All rights reserved.
//

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdio>

#include "../../cuda-dpd/dpd-rng.h"

using namespace std;

const float dt = 0.001;
const float kBT = 0.0945;
const float gammadpd = 45;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = 92.22255689f;//sigma / sqrt(dt);
const float aij = 25;
const float seed = 1.0f;

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        abort();
    }
}

struct Particle
{
    float4 *x, *u, *a;
};


template<int s>
inline __device__ float viscosity_function(float x)
{
    return sqrtf(viscosity_function<s - 1>(x));
}

template<> __device__ inline float viscosity_function<1>(float x) { return sqrtf(x); }
template<> __device__ inline float viscosity_function<0>(float x) { return x; }

__forceinline__ __device__ float4 _dpd_interaction( const int dpid, const float4 xdest, const float4 udest,
        const int spid, const float4 xsrc,  const float4 usrc )
{
    const float _xr = xdest.x - xsrc.x;
    const float _yr = xdest.y - xsrc.y;
    const float _zr = xdest.z - xsrc.z;

    const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

    const float invrij = rsqrtf( rij2 );
    const float rij = rij2 * invrij;
    const float wc = max(0.f, 1 - rij);
    const float wr = viscosity_function < 0 > ( wc );

    const float xr = _xr * invrij;
    const float yr = _yr * invrij;
    const float zr = _zr * invrij;

    const float rdotv =
            xr * ( udest.x - usrc.x ) +
            yr * ( udest.y - usrc.y ) +
            zr * ( udest.z - usrc.z );

    const float myrandnr = Logistic::mean0var1( seed, xmin( spid, dpid ), xmax( spid, dpid ) );

    const float strength = aij * wc - ( gammadpd * wr * rdotv + sigmaf * myrandnr ) * wr;

    return make_float4( strength * xr, strength * yr, strength * zr, 0.0f );
}

__global__ void streamCalc(float4 *  const x1, float4 *  const u1,
        float4 *  const x2, float4 *  const u2,
        float4* a1, float4* a2, int n)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    const float4 xdst = x1[pid];
    const float4 udst = u1[pid];

    const float4 xsrc = x2[pid];
    const float4 usrc = u2[pid];

    const float4 f = _dpd_interaction(pid, xdst, udst, pid + n, xsrc, usrc);

    a1[pid] = f;
    a2[pid] = make_float4(-f.x, -f.y, -f.z, 0.0f);
}

__inline__ __device__ float3 warpReduceSum(float3 val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
        val.z += __shfl_down(val.z, offset);
    }
    return val;
}

#define ndsts 2
__launch_bounds__(128, 8)
__global__ void squareCalc(float4 * const x, float4 * const u, float4* a, int n)
{
    // One warp per ndsts particles

    const uint gid = threadIdx.x + blockDim.x * blockIdx.x;

    const uint dststart = ndsts*(gid >> 5); // / warpSize;
    const uint srcstart = gid & 0x1f; // % warpSize;

    if (dststart >= n) return;

    float3 f[ndsts];
    float4 xdst[ndsts], udst[ndsts];

#pragma unroll
    for (int i=0; i<ndsts; i++)
        f[i] = make_float3(0.0f, 0.0f, 0.0f);

#pragma unroll
    for (int sh = 0; sh < ndsts; sh++)
    {
        if (dststart + sh < n)
        {
            xdst[sh] = x[dststart + sh];
            udst[sh] = u[dststart + sh];
        }
    }

#pragma unroll 1
    for (int src = srcstart; src < n; src += warpSize)
    {
        const float4 xsrc = x[src];
        const float4 usrc = u[src];

#pragma unroll
        for (int sh = 0; sh < ndsts; sh++)
        {
            if (dststart + sh < n)
            {
                const float4 f4 = _dpd_interaction(dststart + sh, xdst[sh], udst[sh], src, xsrc, usrc);
                f[sh].x += f4.x;
                f[sh].y += f4.y;
                f[sh].z += f4.z;
            }
        }
    }

#pragma unroll
    for (int sh = 0; sh < ndsts; sh++)
    {
        if (dststart + sh < n)
        {
            f[sh] = warpReduceSum(f[sh]);
            a[dststart + sh] = make_float4(f[sh].x, f[sh].y, f[sh].z, 0.0f);
        }
    }
}

__forceinline__ __device__ float4 shfl_float4(const float4 x, const uint id)
{
    return make_float4(__shfl(x.x, id), __shfl(x.y, id), __shfl(x.z, id), __shfl(x.w, id));
}

__launch_bounds__(128, 8)
__global__ void squareCalcv2(float4 * const x, float4 * const u, float4* a, int n)
{
    const uint gid = threadIdx.x + blockDim.x * blockIdx.x;
    const uint dststart = gid * ndsts;
    if (dststart >= n) return;
    const uint thid = gid & 0x1f; // % warpSize;

    float3 f[ndsts];
    float4 xdst[ndsts], udst[ndsts];

#pragma unroll
    for (uint i=0; i<ndsts; i++)
        f[i] = make_float3(0.0f, 0.0f, 0.0f);

#pragma unroll
    for (uint sh = 0; sh < ndsts; sh++)
    {
        //if (dststart + sh < n)
        {
            xdst[sh] = x[dststart + sh];
            udst[sh] = u[dststart + sh];
        }
    }

    for (uint i=0; i<n; i+=warpSize)
    {
        const uint src = i + thid;
        const float4 myxsrc = x[src];
        const float4 myusrc = u[src];

#pragma unroll 5
        for (uint j=0; j<32; j++)
        {
            const uint srcThId = (thid + j) & 0x1f;
            const float4 xsrc = shfl_float4(myxsrc, srcThId);
            const float4 usrc = shfl_float4(myusrc, srcThId);

#pragma unroll
            for (uint sh = 0; sh < ndsts; sh++)
            {
                //if (dststart + sh < n)
                {
                    const float4 f4 = _dpd_interaction(i + srcThId, xdst[sh], udst[sh], src, xsrc, usrc);
                    //printf("%d %d\n", i + srcThId, src);
                    f[sh].x += f4.x;
                    f[sh].y += f4.y;
                    f[sh].z += f4.z;
                }
            }
        }
    }

#pragma unroll
    for (int sh = 0; sh < ndsts; sh++)
    {
        if (dststart + sh < n)
        {
            f[sh] = warpReduceSum(f[sh]);
            a[dststart + sh] = make_float4(f[sh].x, f[sh].y, f[sh].z, 0.0f);
        }
    }
}


int main(int argc, const char * argv[])
{
    static const int n = 8*100000;

    Particle p1, p2;
    p1.x = new float4[n];
    p1.u = new float4[n];
    p2.x = new float4[n];
    p2.u = new float4[n];

    cout << "Initializing..." << endl;

#pragma omp parallel for
    for (int i=0; i<n; i++)
    {
        p1.x[i].x = drand48();
        p1.x[i].y = drand48();
        p1.x[i].z = drand48();

        float rij2 = 10;
        do
        {
            p2.x[i].x = drand48();
            p2.x[i].y = drand48();
            p2.x[i].z = drand48();

            const float _xr = p1.x[i].x - p2.x[i].x;
            const float _yr = p1.x[i].y - p2.x[i].y;
            const float _zr = p1.x[i].z - p2.x[i].z;

            rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

        } while (rij2 > 1.0);

        p1.u[i].x = drand48() - 0.5f;
        p1.u[i].y = drand48() - 0.5f;
        p1.u[i].z = drand48() - 0.5f;
        p2.u[i].x = drand48() - 0.5f;
        p2.u[i].y = drand48() - 0.5f;
        p2.u[i].z = drand48() - 0.5f;
    }


    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    Particle dp1, dp2;
    CUDA_CHECK( cudaMalloc(&dp1.x, n * sizeof(float4)) );
    CUDA_CHECK( cudaMalloc(&dp1.u, n * sizeof(float4)) );
    CUDA_CHECK( cudaMalloc(&dp2.x, n * sizeof(float4)) );
    CUDA_CHECK( cudaMalloc(&dp2.u, n * sizeof(float4)) );

    CUDA_CHECK( cudaMalloc(&dp1.a, n * sizeof(float4)) );
    CUDA_CHECK( cudaMalloc(&dp2.a, n * sizeof(float4)) );

    CUDA_CHECK( cudaMemcpy(dp1.x, p1.x, n * sizeof(float4), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dp1.u, p1.u, n * sizeof(float4), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dp2.x, p2.x, n * sizeof(float4), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dp2.u, p2.u, n * sizeof(float4), cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaDeviceSynchronize() );

    cout << "Benchmark will begin now" << endl;

    const int trials = 1;

    //=====================================================================================================
    //=====================================================================================================

    double avg = 0;
    vector<double> records;

    for (int i = 0; i<trials; i++)
    {
        const int th = 128;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        streamCalc<<< (n+th-1)/th, th >>> (dp1.x, dp1.u, dp2.x, dp2.u, dp1.a, dp2.a, n);
        cudaEventRecord(stop);

        CUDA_CHECK( cudaEventSynchronize(stop) );
        CUDA_CHECK( cudaPeekAtLastError() );

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        double inters = (double)n / ms * 1e-3;
        records.push_back(inters);
        avg += inters;
    }

    avg /= trials;
    sort(records.begin(), records.end());

    printf("\nStream of interactions:\n");
    printf("  10%% / mean / 90%%  :  %.3f / %.3f / %.3f  MI/s (max %.3f GB/s)\n",
            records[trials / 10], avg, records[(9*trials) / 10 - 1], 96.0 * records.back() / 1e3);

    //=====================================================================================================
    //=====================================================================================================

    records.clear();
    avg = 0;
    cudaFuncSetCacheConfig(squareCalc, cudaFuncCachePreferL1);
    for (int i = 0; i<trials; i++)
    {
        const int th = 128;
        const int ns = 32*4000;
        const int grid = ((ns*32 + ndsts-1) / ndsts + th-1) / th;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        //squareCalc<<< grid, th >>> (dp1.x, dp1.u, dp1.a, ns);

        squareCalcv2<<< (ns/ndsts + th - 1) / th, th >>> (dp1.x, dp1.u, dp1.a, ns);

        cudaEventRecord(stop);

        CUDA_CHECK( cudaEventSynchronize(stop) );
        CUDA_CHECK( cudaPeekAtLastError() );

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        const double inters = (double)ns*ns / ms * 1e-3;
        records.push_back(inters);
        avg += inters;
    }

    avg /= trials;
    sort(records.begin(), records.end());

    printf("\nAll to all interactions:\n");
    printf("  10%% / mean / 90%%  :  %.3f / %.3f / %.3f  MI/s (max %.3f Gflops/s)\n",
            records[trials / 10], avg, records[(9*trials) / 10 - 1], 89.0 * records.back() / 1e3);

    return 0;
}



