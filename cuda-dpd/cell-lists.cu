/*
 *  cell-lists.cu
 *  Part of uDeviceX/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-07-21.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <unistd.h>

#include "cell-lists.h"

__device__ int encode(int ix, int iy, int iz, int3 ncells)
{
    const int retval = ix + ncells.x * (iy + iz * ncells.y);

    return retval;
}

__device__ int3 decode(int code, int3 ncells)
{
    const int ix = code % ncells.x;
    const int iy = (code / ncells.x) % ncells.y;
    const int iz = (code / ncells.x/ ncells.y);

    return make_int3(ix, iy, iz);
}

#define CC(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        sleep(5);
        if (abort) exit(code);
    }
}

__global__ void pid2code(int * codes, int * pids, const int np, const float * xyzuvw, const int3 ncells, const float3 domainstart, const float invrc)
{
    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= np)
        return;

    const float x = (xyzuvw[0 + 6 * pid] - domainstart.x) * invrc;
    const float y = (xyzuvw[1 + 6 * pid] - domainstart.y) * invrc;
    const float z = (xyzuvw[2 + 6 * pid] - domainstart.z) * invrc;

    int ix = (int)floor(x);
    int iy = (int)floor(y);
    int iz = (int)floor(z);

    /*   if( !(ix >= 0 && ix < ncells.x) ||
         !(iy >= 0 && iy < ncells.y) ||
         !(iz >= 0 && iz < ncells.z))
         printf("pid %d: oops %f %f %f -> %d %d %d\n", pid, x, y, z, ix, iy, iz);
     */

    ix = max(0, min(ncells.x - 1, ix));
    iy = max(0, min(ncells.y - 1, iy));
    iz = max(0, min(ncells.z - 1, iz));

    codes[pid] = encode(ix, iy, iz, ncells);
    pids[pid] = pid;
};

__global__ void _gather(const float * input, const int * indices, float * output, const int n)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < n)
        output[tid] = input[(tid % 6) + 6 * indices[tid / 6]];
}

__global__ void _generate_cids(int * cids, const int ntotcells, const int offset, const int3 ncells)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < ntotcells)
    {
        const int xcid = tid % ncells.x;
        const int ycid = (tid / ncells.x) % ncells.y;
        const int zcid = (tid / ncells.x / ncells.y) % ncells.z;

        cids[tid] = encode(xcid, ycid, zcid, ncells) + offset;
    }
}

    __global__
void _count_particles(const int * const cellsstart, int * const cellscount, const int ncells)
{
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < ncells)
        cellscount[tid] -= cellsstart[tid];
}


struct is_gzero
{
    __host__ __device__
        bool operator()(const int &x)
        {
            return  x > 0;
        }
};

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

using namespace thrust;

template<typename T> T * _ptr(device_vector<T>& v) { return raw_pointer_cast(v.data()); }

void build_clists_vanilla(float * const xyzuvw, int np, const float rc,
        const int xcells, const int ycells, const int zcells,
        const float xstart, const float ystart, const float zstart,
        int * const order, int * cellsstart, int * cellscount, std::pair<int, int *> * nonemptycells, cudaStream_t stream, const float * const src_device_xyzuvw)
{
    device_vector<int> codes(np), pids(np);
    pid2code<<<(np + 127) / 128, 128>>>(_ptr(codes), _ptr(pids), np, xyzuvw, make_int3(xcells, ycells, zcells), make_float3(xstart, ystart, zstart), 1./rc);

    sort_by_key(codes.begin(), codes.end(), pids.begin());

    {
        device_vector<float> tmp(np * 6);

        if (src_device_xyzuvw)
            copy(device_ptr<float>((float *)src_device_xyzuvw), device_ptr<float>((float *)src_device_xyzuvw + 6 * np), tmp.begin());
        else
            copy(device_ptr<float>(xyzuvw), device_ptr<float>(xyzuvw + 6 * np), tmp.begin());

        _gather<<<(6 * np + 127) / 128, 128>>>(_ptr(tmp), _ptr(pids), xyzuvw, 6 * np);
        CC(cudaPeekAtLastError());
    }

    const int ncells = xcells * ycells * zcells;
    device_vector<int> cids(ncells), cidsp1(ncells);

    _generate_cids<<< (cids.size() + 127) / 128, 128>>>(_ptr(cids), ncells, 0,  make_int3(xcells, ycells, zcells));
    _generate_cids<<< (cidsp1.size() + 127) / 128, 128>>>(_ptr(cidsp1), ncells, 1, make_int3(xcells, ycells, zcells) );

    lower_bound(codes.begin(), codes.end(), cids.begin(), cids.end(), device_ptr<int>(cellsstart));
    lower_bound(codes.begin(), codes.end(), cidsp1.begin(), cidsp1.end(), device_ptr<int>(cellscount));

    _count_particles<<<(ncells + 127) / 128, 128>>> (cellsstart, cellscount, ncells);

    if (nonemptycells != NULL)
    {
        const int nonempties = copy_if(counting_iterator<int>(0), counting_iterator<int>(ncells),
                device_ptr<int>(cellscount), device_ptr<int>(nonemptycells->second), is_gzero())
            - device_ptr<int>(nonemptycells->second);

        nonemptycells->first = nonempties;
    }

    if (order != NULL)
        copy(pids.begin(), pids.end(), order);
}

