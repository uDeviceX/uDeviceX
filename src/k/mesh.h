namespace k_mesh
{
    #define _HD_ __host__ __device__

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

    __global__ void inside(const Particle *pp, const int n, const float *vv, const int *tt, const int nt, /**/ int *inout)
    {
        const int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if (gid >= n) return;

        int count = 0;

        const Particle p = pp[gid];
        const float origin[3] = {0, 0, 0};
        
        for (int i = 0; i < nt; ++i)
        {
            const int t1 = tt[3*i + 0];
            const int t2 = tt[3*i + 1];
            const int t3 = tt[3*i + 2];

            const float a[3] = {vv[3*t1 + 0], vv[3*t1 + 1], vv[3*t1 + 2]};
            const float b[3] = {vv[3*t2 + 0], vv[3*t2 + 1], vv[3*t2 + 2]};
            const float c[3] = {vv[3*t3 + 0], vv[3*t3 + 1], vv[3*t3 + 2]};
            
            if (in_tetrahedron(p.r, a, b, c, origin)) ++count;
        }
        
        inout[gid] = count % 2;
    }

    static __inline__ __device__ uint umin(uint a, uint b) {return a > b ? a : b;}
    static __inline__ __device__ uint umax(uint a, uint b) {return a > b ? a : b;}
    
    static __device__ uint3 warpReduceMin(uint3 val)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            val.x = umin(val.x, __shfl_down(val.x, offset));
            val.y = umin(val.y, __shfl_down(val.y, offset));
            val.z = umin(val.z, __shfl_down(val.z, offset));
        }
        return val;
    }
    
    static __device__ uint3 warpReduceMax(uint3 val)
    {
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            val.x = umax(val.x, __shfl_down(val.x, offset));
            val.y = umax(val.y, __shfl_down(val.y, offset));
            val.z = umax(val.z, __shfl_down(val.z, offset));
        }
        return val;
    }
    
    // http://stereopsis.com/radix.html
    // we need to convert floats to uint32 because atomicMin/Max
    // is only available for integers
    
    static __device__ uint code(uint f)
    {
        uint mask = -int(f >> 31) | 0x80000000;
        return f ^ mask;
    }

    static __device__ uint decode(uint f)
    {
        uint mask = ((f >> 31) - 1) | 0x80000000;
        return f ^ mask;
    }

#define LB 0x00000000
#define UB 0xFFFFFFFF
    
    __global__ void init_bboxes(const int ns, /**/ uint2 *bboxes)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < 3 * ns)
        {
            uint2 cb = make_uint2(UB, LB);
            cb.x = code(cb.x);
            cb.y = code(cb.y);
            bboxes[i] = cb;
        }
    }
    
    __global__ void bboxes(const uint *pp, const int nps, const int ns, /**/ uint *bboxes)
    {
        uint3 mins = make_uint3(UB, UB, UB);
        uint3 maxs = make_uint3(LB, LB, LB);
        
        const int mid = blockIdx.y; // mesh Id

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nps; i += blockIdx.x * blockDim.x)
        {
            const int base = 6 * (mid * nps + i);
            
            const uint x = code(pp[base + X]);
            const uint y = code(pp[base + Y]);
            const uint z = code(pp[base + Z]);

            mins.x = umin(x, mins.x); mins.y = umin(y, mins.y); mins.z = umin(z, mins.z);
            maxs.x = umax(x, maxs.x); maxs.y = umax(y, maxs.y); maxs.z = umax(z, maxs.z);
        }

        mins = warpReduceMin(mins);
        maxs = warpReduceMax(maxs);

        mins.x = code(mins.x); mins.y = code(mins.y); mins.z = code(mins.z);
        maxs.x = code(maxs.x); maxs.y = code(maxs.y); maxs.z = code(maxs.z);
        
        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicMin(bboxes + 6 * mid + 0, mins.x);
            atomicMax(bboxes + 6 * mid + 1, maxs.x);
            
            atomicMin(bboxes + 6 * mid + 2, mins.y);
            atomicMax(bboxes + 6 * mid + 3, maxs.y);

            atomicMin(bboxes + 6 * mid + 4, mins.z);
            atomicMax(bboxes + 6 * mid + 5, maxs.z);
        }
    }

    __global__ void decode_bboxes(const int ns, /**/ uint2 *bboxes)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < 3 * ns)
        {
            uint2 cb = bboxes[i];
            cb.x = decode(cb.x);
            cb.y = decode(cb.y);
            bboxes[i] = cb;
        }
    }
#undef LB
#undef LB
}
