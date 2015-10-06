/*
 *  velcontroller.cu
 *  ctc falcon
 *
 *  Created by Dmitry Alexeev on Sep 24, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#include "velcontroller.h"
#include "helper_math.h"

namespace VelControlKernels
{
    struct CellInfo
    {
        uint cellsx, cellsy, cellsz;
        uint xl[3];
        uint n[3];
    };

    __constant__ CellInfo info;

    __global__ void sample(const int * const __restrict__ cellsstart, const Particle* const __restrict__ p, float3* res)
    {
        const uint3 ccoos = {threadIdx.x + blockIdx.x*blockDim.x,
                threadIdx.y + blockIdx.y*blockDim.y,
                threadIdx.z + blockIdx.z*blockDim.z};

        if (ccoos.x < info.n[0] && ccoos.y < info.n[1] && ccoos.z < info.n[2])
        {
            const uint cid = (ccoos.x + info.xl[0]) * info.cellsy * info.cellsz + (ccoos.y + info.xl[1]) * info.cellsz + (ccoos.z + info.xl[2]);
            const uint resid = ccoos.x * info.n[1] * info.n[1] + ccoos.y * info.n[2] + ccoos.z;

            for (uint pid = cellsstart[cid]; pid < cellsstart[cid+1]; pid++)
            {
                res[resid].x += p[pid].u[0];
                res[resid].y += p[pid].u[1];
                res[resid].z += p[pid].u[2];
            }
        }
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

    __global__ void reduceByWarp(float3 *res, const float3 * const __restrict__ vel, const uint total)
    {
        assert(blockDim.x == 32);
        const uint id = threadIdx.x + blockIdx.x*blockDim.x;
        const uint ch = blockIdx.x;
        if (id >= total) return;

        const float3 val  = vel[id];
        const float3 rval = warpReduceSum(val);

        if ((threadIdx.x % warpSize) == 0)
            res[ch]=rval;
    }
}

VelController::VelController(int xl[3], int xh[3], int mpicoos[3], float3 desired, MPI_Comm comm) :
        desired(desired), Kp(0.003), Ki(0.001), Kd(0.1), sampleid(0)
{
    MPI_CHECK( MPI_Comm_dup(comm, &this->comm) );
    MPI_CHECK( MPI_Comm_size(comm, &size) );
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );
    const int L[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};

    for (int d=0; d<3; d++)
    {
        myxl[d] = max(xl[d], L[d]*mpicoos[d]    ) % (L[d]+1) - 0.5*L[d];
        myxh[d] = min(xh[d], L[d]*(mpicoos[d]+1)) % (L[d]+1) - 0.5*L[d];

        n[d] = myxh[d] - myxl[d];
    }

    if (n[0] > 0 && n[1] > 0 && n[2] > 0)
        total = n[0] * n[1] * n[2];
    else
        total = 0;

    MPI_CHECK( MPI_Allreduce(&total, &globtot, 1, MPI_INT, MPI_SUM, comm) );

    vel.resize(total);
    if (total)
        CUDA_CHECK( cudaMemset(vel.data, 0, n[0] * n[1] * n[2] * sizeof(float3)) );

    VelControlKernels::CellInfo info = {L[0], L[1], L[2], {xl[0], xl[1], xl[2]}, {n[0], n[1], n[2]}};
    CUDA_CHECK( cudaMemcpyToSymbol(VelControlKernels::info, &info, sizeof(info)) );

    s = make_float3(0, 0, 0);
    old = desired;
}

void VelController::sample(const int * const cellsstart, const Particle* const p, cudaStream_t stream)
{
    dim3 block(8, 4, 4);
    dim3 grid(  (n[0] + block.x - 1) / block.x,
            (n[1] + block.y - 1) / block.y,
            (n[2] + block.z - 1) / block.z );

    sampleid++;
    if (total)
        VelControlKernels::sample <<<grid, block, 0, stream>>> (cellsstart, p, vel.data);
}

float3 VelController::adjustF(cudaStream_t stream)
{
    const int chunks = (total+31) / 32;
    if (avgvel.size < chunks) avgvel.resize(chunks);

    if (total)
    {
        VelControlKernels::reduceByWarp <<< (total + 31) / 32, 32, 0, stream >>> (avgvel.devptr, vel.data, total);
        CUDA_CHECK( cudaStreamSynchronize(stream) );
    }

    float3 cur = make_float3(0, 0, 0);
    for (int i=0; i<chunks; i++)
        cur += avgvel.data[i];

    MPI_CHECK( MPI_Allreduce(MPI_IN_PLACE, &cur.x, 3, MPI_FLOAT, MPI_SUM, comm) );
    cur /= globtot * sampleid;

    float3 err = desired - cur;
    float3 de  = err - old;
    s += err;
    float3 f = Kp*err + Ki*s + Kd*de;

    if (total)
        CUDA_CHECK( cudaMemset(vel.data, 0, n[0] * n[1] * n[2] * sizeof(float3)) );
    sampleid = 0;

    if (rank==0) printf("Vel:  [%8.3f  %8.3f  %8.3f], force: [%8.3f  %8.3f  %8.3f]\n", cur.x, cur.y, cur.z,  f.x, f.y, f.z);

    return f;
}

