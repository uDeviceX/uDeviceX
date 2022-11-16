#include <stdio.h>


#include "inc/conf.h"

#include "inc/def.h"
#include "utils/msg.h"
#include "d/q.h"
#include "d/ker.h"
#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "algo/utils/shfl.h"
#include "algo/utils/dev.h"

#include "utils/kl.h"

#include "struct/particle/dev.h"

#include "imp.h"

enum {
    WARPSIZE   = 32,
    BLOCK_SIZE = 128,
    NWARPS     = BLOCK_SIZE / WARPSIZE
};

namespace minmax_dev {
static const float MINV = -100000000.;
static const float MAXV =  100000000.;

static __device__ float3 minf3(float3 a, float3 b) {
    return make_float3(min(a.x, b.x),
                       min(a.y, b.y),
                       min(a.z, b.z));
}

static __device__ float3 maxf3(float3 a, float3 b) {
    return make_float3(max(a.x, b.x),
                       max(a.y, b.y),
                       max(a.z, b.z));
}

static __device__ float3 warpReduce_minf3(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x = min(val.x, shfl_down(val.x, offset));
        val.y = min(val.y, shfl_down(val.y, offset));
        val.z = min(val.z, shfl_down(val.z, offset));
    }
    return val;
}

static __device__ float3 warpReduce_maxf3(float3 val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x = max(val.x, shfl_down(val.x, offset));
        val.y = max(val.y, shfl_down(val.y, offset));
        val.z = max(val.z, shfl_down(val.z, offset));
    }
    return val;
}


__global__ void minmax(int nv, const Particle *pp, float3 *lo, float3 *hi) {
    enum {LO, HI, D};
    int i, objid, stride, pid, wid, lid;
    float3 r, lmin, lmax;
    __shared__ float3 lohi[D][NWARPS];
        
    stride = blockDim.x * gridDim.x;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    lid = i % warpSize;
    wid = i / warpSize;
    objid = blockIdx.y;

    lmin = make_float3(MAXV, MAXV, MAXV);
    lmax = make_float3(MINV, MINV, MINV);
    
    for (; i < nv; i += stride) {
        pid = objid * nv + i;
        r = fetch_pos(pid, pp);
        lmin = minf3(r, lmin);
        lmax = maxf3(r, lmax);
    }

    lmin = warpReduce_minf3(lmin);
    lmax = warpReduce_maxf3(lmax);

    if (lid == 0) {
        lohi[LO][wid] = lmin;
        lohi[HI][wid] = lmax;
    }
    
    __syncthreads();
    
    if (wid == 0) {
        if (lid < NWARPS) {
            lmin = lohi[LO][lid];
            lmax = lohi[HI][lid];
        }
        lmin = warpReduce_minf3(lmin);
        lmax = warpReduce_maxf3(lmax);

        if (lid == 0) {
            lo[objid] = lmin;
            hi[objid] = lmax;
        }
    }
}
}

void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi) {
    dim3 thrd(BLOCK_SIZE, 1);
    dim3 blck(1, nobj);
    
    KL(minmax_dev::minmax, (blck, thrd), (nv, pp, lo, hi));
}
