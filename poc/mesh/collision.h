#include <cstdio>
#include <cstdlib>

#pragma once
#define _HD_ __host__ __device__

namespace collision
{
    enum {X, Y, Z};

    static _HD_ bool same_side(const float *x, const float *p, const float *a, const float *b, const float *inplane)
    {
        const float n[3] = {a[Y] * b[Z] - a[Z] * b[Y],
                            a[Z] * b[X] - a[X] * b[Z],
                            a[X] * b[Y] - a[Y] * b[X]};

        const float ndx = n[X] * (x[X] - inplane[X]) + n[Y] * (x[Y] - inplane[Y]) + n[Z] * (x[Z] - inplane[Z]);
        const float ndp = n[X] * (p[X] - inplane[X]) + n[Y] * (p[Y] - inplane[Y]) + n[Z] * (p[Z] - inplane[Z]);
        
        return ndx * ndp > 0;
    }
    
    _HD_ bool in_tetrahedron(const float *x, const float *A, const float *B, const float *C, const float *D)
    {
        const float AB[3] = {B[X] - A[X], B[Y] - A[Y], B[Z] - A[Z]};
        const float AC[3] = {C[X] - A[X], C[Y] - A[Y], C[Z] - A[Z]};
        const float AD[3] = {D[X] - A[X], D[Y] - A[Y], D[Z] - A[Z]};
    
        const float BC[3] = {C[X] - B[X], C[Y] - B[Y], C[Z] - B[Z]};
        const float BD[3] = {D[X] - B[X], D[Y] - B[Y], D[Z] - B[Z]};

        return
            same_side(x, A, BC, BD, D) &&
            same_side(x, B, AD, AC, D) &&
            same_side(x, C, AB, BD, D) &&
            same_side(x, D, AB, AC, A);
    }

    _HD_ int in_mesh_1p(const float *r, const float *vv, const int *tt, const int nt)
    {
        int c = 0;

        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int *t = tt + 3 * i;

            if (in_tetrahedron(r, vv + 3*t[0], vv + 3*t[1], vv + 3*t[2], origin)) ++c;
        }
        
        return (c+1)%2;
    }

    void in_mesh(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        for (int i = 0; i < n; ++i)
        inout[i] = in_mesh_1p(rr + 3*i, vv, tt, nt);
    }

    __global__ void in_mesh_k(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        const int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid >= n) return;

        int count = 0;
        const float r[3] = {rr[3*gid + 0], rr[3*gid + 1], rr[3*gid + 2]};
        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int t1 = tt[3*i + 0];
            const int t2 = tt[3*i + 1];
            const int t3 = tt[3*i + 2];

            const float a[3] = {vv[3*t1 + 0], vv[3*t1 + 1], vv[3*t1 + 2]};
            const float b[3] = {vv[3*t2 + 0], vv[3*t2 + 1], vv[3*t2 + 2]};
            const float c[3] = {vv[3*t3 + 0], vv[3*t3 + 1], vv[3*t3 + 2]};
                
            if (in_tetrahedron(r, a, b, c, origin)) ++count;
        }
        
        inout[gid] = (count+1)%2;
    }

    #define NTSHARED 128
    __global__ void in_mesh_k_shared(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        __shared__ float _sdata[9*NTSHARED];
        
        const int gid = threadIdx.x + blockIdx.x * blockDim.x;
        
        int count = 0;

        float r[3] = {0};
        if (gid < n)
        {
            r[0] = rr[3*gid + 0];
            r[1] = rr[3*gid + 1];
            r[2] = rr[3*gid + 2];
        }

        const float origin[3] = {0, 0, 0};

        for (int tb = 0; tb < (nt + NTSHARED-1)/NTSHARED; ++tb)
        {
            __syncthreads();
            
            // load triangles in shared mem
            if (threadIdx.x < NTSHARED)
            {
                const int tid = tb*NTSHARED + threadIdx.x;

                if (tid < nt)
                {
                    const int t1 = tt[3*tid + 0];
                    const int t2 = tt[3*tid + 1];
                    const int t3 = tt[3*tid + 2];

                    const int base = 9 * threadIdx.x;

                    _sdata[base + 0] = vv[3*t1 + 0];
                    _sdata[base + 1] = vv[3*t1 + 1];
                    _sdata[base + 2] = vv[3*t1 + 2];

                    _sdata[base + 3] = vv[3*t2 + 0];
                    _sdata[base + 4] = vv[3*t2 + 1];
                    _sdata[base + 5] = vv[3*t2 + 2];

                    _sdata[base + 6] = vv[3*t3 + 0];
                    _sdata[base + 7] = vv[3*t3 + 1];
                    _sdata[base + 8] = vv[3*t3 + 2];
                }
            }

            __syncthreads();

            if (gid < n)
            {
                const int max = (tb + 1) * NTSHARED <= nt ? NTSHARED : nt % NTSHARED;

                if (gid == 100000)
                printf("%d %d\n", tb, max);
                
                // perform computation on these triangles
                for (int i = 0; i < max; ++i)
                {
                    const int base = 9 * i;
                    const float a[3] = {_sdata[base + 0], _sdata[base + 1], _sdata[base + 2]};
                    const float b[3] = {_sdata[base + 3], _sdata[base + 4], _sdata[base + 5]};
                    const float c[3] = {_sdata[base + 6], _sdata[base + 7], _sdata[base + 8]};
                
                    if (in_tetrahedron(r, a, b, c, origin)) ++count;
                }
            }
        }
        
        if (gid < n) inout[gid] = (count+1)%2;
    }

    void in_mesh_dev(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        in_mesh_k <<< (127 + n) / 128, 128 >>>(rr, n, vv, tt, nt, /**/ inout);
        //in_mesh_k_shared <<< (127 + n) / 128, 128 >>>(rr, n, vv, tt, nt, /**/ inout);
    }
}
