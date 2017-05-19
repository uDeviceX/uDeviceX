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
        
        return (c%2) ? 0 : -1;
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
        
        inout[gid] = (count%2) ? 0 : -1;
    }

    __global__ void in_mesh_kt(const float *rr, const int n, cudaTextureObject_t vv, cudaTextureObject_t tt, const int nt, /**/ int *inout)
    {
        const int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid >= n) return;

        int count = 0;
        const float r[3] = {rr[3*gid + 0], rr[3*gid + 1], rr[3*gid + 2]};
        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int t1 = tex1Dfetch<int>(tt, 3*i + 0);
            const int t2 = tex1Dfetch<int>(tt, 3*i + 1);
            const int t3 = tex1Dfetch<int>(tt, 3*i + 2);

#define ld1(t, a) tex1Dfetch<float>(vv, 3*t + a)
#define ld3(t) {ld1(t, 0), ld1(t, 1), ld1(t, 2)}

            const float a[3] = ld3(t1);
            const float b[3] = ld3(t2);
            const float c[3] = ld3(t3);
#undef ld1
#undef ld3
            if (in_tetrahedron(r, a, b, c, origin)) ++count;
        }
        
        inout[gid] = (count%2) ? 0 : -1;
    }

    #define NTSHARED 64
    __global__ void in_mesh_ks(const float *rr, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
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
        
        if (gid < n) inout[gid] = (count%2) ? 0 : -1;
    }

    __global__ void in_mesh_kts(const float *rr, const int n, cudaTextureObject_t vv, cudaTextureObject_t tt, const int nt, /**/ int *inout)
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
                    const int t1 = tex1Dfetch<int>(tt, 3*tid + 0);
                    const int t2 = tex1Dfetch<int>(tt, 3*tid + 1);
                    const int t3 = tex1Dfetch<int>(tt, 3*tid + 2);

                    const int base = 9 * threadIdx.x;

                    _sdata[base + 0] = tex1Dfetch<float>(vv, 3*t1 + 0);
                    _sdata[base + 1] = tex1Dfetch<float>(vv, 3*t1 + 1);
                    _sdata[base + 2] = tex1Dfetch<float>(vv, 3*t1 + 2);

                    _sdata[base + 3] = tex1Dfetch<float>(vv, 3*t2 + 0);
                    _sdata[base + 4] = tex1Dfetch<float>(vv, 3*t2 + 1);
                    _sdata[base + 5] = tex1Dfetch<float>(vv, 3*t2 + 2);

                    _sdata[base + 6] = tex1Dfetch<float>(vv, 3*t3 + 0);
                    _sdata[base + 7] = tex1Dfetch<float>(vv, 3*t3 + 1);
                    _sdata[base + 8] = tex1Dfetch<float>(vv, 3*t3 + 2);
                }
            }

            __syncthreads();

            if (gid < n)
            {
                const int max = (tb + 1) * NTSHARED <= nt ? NTSHARED : nt % NTSHARED;
                
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
        
        if (gid < n) inout[gid] = (count%2) ? 0 : -1;
    }

    //#define VANILLA
    //#define SHARED
    //#define TEXTURE
#define TEXTURE_SHARED
    
    void in_mesh_dev(const float *rr, const int n, float *vv, const int nv, int *tt, const int nt, /**/ int *inout)
    {
#if defined (VANILLA)
        
        in_mesh_k <<< (127 + n) / 128, 128 >>>(rr, n, vv, tt, nt, /**/ inout);
        
#elif defined (SHARED)
        
        in_mesh_ks <<< (127 + n) / 128, 128 >>>(rr, n, vv, tt, nt, /**/ inout);
        
#elif defined (TEXTURE) || defined (TEXTURE_SHARED)

        cudaTextureObject_t vvt, ttt;
        cudaResourceDesc    resDv, resDt;
        cudaTextureDesc     texDv, texDt;

        memset(&resDv, 0, sizeof(resDv));
        resDv.resType = cudaResourceTypeLinear;
        resDv.res.linear.devPtr  = vv;
        resDv.res.linear.sizeInBytes = 3 * sizeof(float) * nv;
        resDv.res.linear.desc = cudaCreateChannelDesc<float>();

        memset(&texDv, 0, sizeof(texDv));
        texDv.normalizedCoords = 0;
        texDv.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&vvt, &resDv, &texDv, NULL);
        
        memset(&resDt, 0, sizeof(resDt));
        resDt.resType = cudaResourceTypeLinear;
        resDt.res.linear.devPtr  = tt;
        resDt.res.linear.sizeInBytes = 3 * sizeof(int) * nt;
        resDt.res.linear.desc = cudaCreateChannelDesc<int>();

        memset(&texDt, 0, sizeof(texDt));
        texDt.normalizedCoords = 0;
        texDt.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&ttt, &resDt, &texDt, NULL);

#if defined (TEXTURE)
        in_mesh_kt <<< (127 + n) / 128, 128 >>>(rr, n, vvt, ttt, nt, /**/ inout);
#elif defined (TEXTURE_SHARED)
        in_mesh_kts <<< (127 + n) / 128, 128 >>>(rr, n, vvt, ttt, nt, /**/ inout);
#endif
#endif
    }
}
