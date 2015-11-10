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
#include <omp.h>

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
    float *x[3], *u[3];
};

struct Acceleration
{
    float *a[3];
};

template<int s>
inline __device__ float viscosity_function(float x)
{
    return sqrtf(viscosity_function<s - 1>(x));
}

template<> __device__ inline float viscosity_function<1>(float x) { return sqrtf(x); }
template<> __device__ inline float viscosity_function<0>(float x) { return x; }

//=====================================================================================================
//=====================================================================================================

__forceinline__ __device__ float3 _dpd_interaction( const int dpid, const float3 xdest, const float3 udest,
        const int spid, const float3 xsrc, const float3 usrc )
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

    return make_float3( strength * xr, strength * yr, strength * zr);
}

__forceinline__ __device__ float4 _dpd_interaction( const int dpid, const float4 xdest, const float4 udest,
        const int spid, const float4 xsrc, const float4 usrc )
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

//=====================================================================================================
// Stream
//=====================================================================================================

#define LOAD4(VAR) \
        const float4 VAR##VAR##1_1 = ((float4*)VAR##1)[id1]; \
        const float4 VAR##VAR##1_2 = ((float4*)VAR##2)[id1]; \
//        const float4 VAR##VAR##2_1 = ((float4*)VAR##1)[id2]; \
//        const float4 VAR##VAR##2_2 = ((float4*)VAR##2)[id2];

#define COMP1(ID, COO) \
        const float3 f##ID = _dpd_interaction(id1*4+ID, \
                make_float3(xx1_1.COO, yy1_1.COO, zz1_1.COO), \
                make_float3(uu1_1.COO, vv1_1.COO, ww1_1.COO), \
                id1*4+ID+n, \
                make_float3(xx1_2.COO, yy1_2.COO, zz1_2.COO), \
                make_float3(uu1_2.COO, vv1_2.COO, ww1_2.COO));

#define COMP2(ID, COO) \
        const float3 f##ID = _dpd_interaction(id2*4+ID-4, \
                make_float3(xx2_1.COO, yy2_1.COO, zz2_1.COO), \
                make_float3(uu2_1.COO, vv2_1.COO, ww2_1.COO), \
                id2*4+ID+n-4, \
                make_float3(xx2_2.COO, yy2_2.COO, zz2_2.COO), \
                make_float3(uu2_2.COO, vv2_2.COO, ww2_2.COO));

#define WRITE1(COO) \
        ((float4*)a##COO##1)[id1] = make_float4( f0.COO,  f1.COO,  f2.COO,  f3.COO); \
        ((float4*)a##COO##2)[id1] = make_float4(-f0.COO, -f1.COO, -f2.COO, -f3.COO);

#define WRITE2(COO) \
        ((float4*)a##COO##1)[id2] = make_float4( f4.COO,  f5.COO,  f6.COO,  f7.COO); \
        ((float4*)a##COO##2)[id2] = make_float4(-f4.COO, -f5.COO, -f6.COO, -f7.COO);

#define streamNdsts 1
//__launch_bounds__(128, 7)
__global__ void streamCalc(float * const x1, float * const y1, float * const z1,
        float * const x2, float * const y2, float * const z2,
        float * const u1, float * const v1, float * const w1,
        float * const u2, float * const v2, float * const w2,
        float * const ax1, float * const ay1, float * const az1,
        float * const ax2, float * const ay2, float * const az2, int n)
{
    const int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= n / (4*streamNdsts) ) return;

    const int id1 = gid;
//    const int id2 = gid + n / (4*streamNdsts);

    LOAD4(x)
    LOAD4(y)
    LOAD4(z)
    LOAD4(u)
    LOAD4(v)
    LOAD4(w)

    COMP1(0, x);
    COMP1(1, y);
    COMP1(2, z);
    COMP1(3, w);

//     COMP2(4, x);
//     COMP2(5, y);
//     COMP2(6, z);
//     COMP2(7, w);

    WRITE1(x);
    WRITE1(y);
    WRITE1(z);
//     WRITE2(x);
//     WRITE2(y);
//     WRITE2(z);
}

//=====================================================================================================
// Square
//=====================================================================================================

#define ndsts 2
__forceinline__ __device__ float4 shfl_float4(const float4 x, const uint id)
{
    return make_float4(__shfl(x.x, id), __shfl(x.y, id), __shfl(x.z, id), 0.0f);
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
        //if (dststart + sh < n)
        {
            f[sh] = warpReduceSum(f[sh]);
            a[dststart + sh] = make_float4(f[sh].x, f[sh].y, f[sh].z, 0.0f);
        }
    }
}


int main(int argc, const char * argv[])
{
    static const int n = 8*5000000;

    Particle p1, p2;
    p1.x[0] = new float[n];
    p1.x[1] = new float[n];
    p1.x[2] = new float[n];
    p1.u[0] = new float[n];
    p1.u[1] = new float[n];
    p1.u[2] = new float[n];

    p2.x[0] = new float[n];
    p2.x[1] = new float[n];
    p2.x[2] = new float[n];
    p2.u[0] = new float[n];
    p2.u[1] = new float[n];
    p2.u[2] = new float[n];

    printf("Initializing...\n");

    //srand48(time(NULL));

    unsigned long t0 = time(NULL);
#pragma omp parallel
    {
        unsigned short xi[3];
        xi[0] = 1;
        xi[1] = 1;
        xi[2] = omp_get_thread_num() + t0;

#pragma omp for
        for (int i=0; i<n; i++)
        {
            p1.x[0][i] = erand48(xi);
            p1.x[1][i] = erand48(xi);
            p1.x[2][i] = erand48(xi);
            p2.x[0][i] = erand48(xi);
            p2.x[1][i] = erand48(xi);
            p2.x[2][i] = erand48(xi);

            p1.u[0][i] = erand48(xi) - 0.5f;
            p1.u[1][i] = erand48(xi) - 0.5f;
            p1.u[2][i] = erand48(xi) - 0.5f;
            p2.u[0][i] = erand48(xi) - 0.5f;
            p2.u[1][i] = erand48(xi) - 0.5f;
            p2.u[2][i] = erand48(xi) - 0.5f;
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    float *x1, *y1, *z1, *x2, *y2, *z2,
    *u1, *v1, *w1, *u2, *v2, *w2,
    *ax1, *ay1, *az1,
    *ax2, *ay2, *az2;

    CUDA_CHECK( cudaMalloc(&x1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&y1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&z1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&x2, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&y2, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&z2, n * sizeof(float)) );

    CUDA_CHECK( cudaMalloc(&u1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&v1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&w1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&u2, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&v2, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&w2, n * sizeof(float)) );

    CUDA_CHECK( cudaMalloc(&ax1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&ay1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&az1, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&ax2, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&ay2, n * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&az2, n * sizeof(float)) );

    CUDA_CHECK( cudaMemcpy(x1, p1.x[0], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(y1, p1.x[1], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(z1, p1.x[2], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(x2, p2.x[0], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(y2, p2.x[1], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(z2, p2.x[2], n * sizeof(float), cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaMemcpy(u1, p1.u[0], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(v1, p1.u[1], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(w1, p1.u[2], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(u2, p2.u[0], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(v2, p2.u[1], n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(w2, p2.u[2], n * sizeof(float), cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaDeviceSynchronize() );

    cout << "Benchmark will begin now" << endl;

    const int trials = 5;

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
        streamCalc<<< (n/4+th-1)/th, th >>> (x1, y1, z1, x2, y2, z2,
                u1, v1, w1, u2, v2, w2,
                ax1, ay1, az1, ax2, ay2, az2, n);
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
            records[trials / 10], avg, records[(9*trials) / 10 - 1], 72.0 * records.back() / 1e3);

    //=====================================================================================================
    //=====================================================================================================

    const int ns = 32*10000;
    struct { float4 *x, *u, *a; } p, dp;
    p.x = new float4[ns];
    p.u = new float4[ns];

#pragma omp parallel
    {
        unsigned short xi[3];
        xi[0] = 1;
        xi[1] = 1;
        xi[2] = omp_get_thread_num() + t0;

#pragma omp for
        for (int i=0; i<ns; i++)
        {
            p.x[i].x = erand48(xi);
            p.x[i].y = erand48(xi);
            p.x[i].z = erand48(xi);

            p.u[i].x = erand48(xi) - 0.5f;
            p.u[i].y = erand48(xi) - 0.5f;
            p.u[i].z = erand48(xi) - 0.5f;
        }
    }

    CUDA_CHECK( cudaMalloc(&dp.x, ns * sizeof(float4)) );
    CUDA_CHECK( cudaMalloc(&dp.u, ns * sizeof(float4)) );
    CUDA_CHECK( cudaMalloc(&dp.a, ns * sizeof(float4)) );

    CUDA_CHECK( cudaMemcpy(dp.x, p.x, ns * sizeof(float4), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dp.u, p.u, ns * sizeof(float4), cudaMemcpyHostToDevice) );

    records.clear();
    avg = 0;
    cudaFuncSetCacheConfig(squareCalcv2, cudaFuncCachePreferL1);
    for (int i = 0; i<trials; i++)
    {
        const int th = 128;
        const int grid = ((ns*32 + ndsts-1) / ndsts + th-1) / th;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        squareCalcv2<<< (ns/ndsts + th - 1) / th, th >>> (dp.x, dp.u, dp.a, ns);
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



