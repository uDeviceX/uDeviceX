#include <limits>
#include <stdint.h>
#include <stdio.h>
#include "rnd.h"
#include "common.h"
#include "common.cuda.h"
#include "inc/type.h"
#include "dpd/local.h"
#include <conf.h>
#include "forces.h"

#include "dpd/dev/float.h"
#include "dpd/imp/type.h"

#include "dpd/dev/decl.h"
#include "dpd/imp/decl.h"

__device__ void f2tof3(float4 r, /**/ float3 *l) { /* lhs = rhs */
    l->x = r.x; l->y = r.y; l->z = r.z;
}

__device__ float3 _dpd_interaction(int dpid, float4 rdest, float4 udest, float4 rsrc, float4 usrc, int spid) {
    float rnd;
    float3 r1, r2, v1, v2;
    float3 f;
  
    rnd = rnd::mean0var1ii( info.seed, xmin( spid, dpid ), xmax( spid, dpid ) );
    f2tof3(rdest, &r1); f2tof3(rsrc, &r2);
    f2tof3(udest, &v1); f2tof3(usrc, &v2);
  
    f = forces::dpd(SOLVENT_TYPE, SOLVENT_TYPE, r1, r2, v1, v2, rnd);
    return f;
}

#define __IMOD(x,y) ((x)-((x)/(y))*(y))

__inline__ __device__ uint __lanemask_lt()
{
    uint mask;
    asm( "mov.u32 %0, %lanemask_lt;" : "=r"( mask ) );
    return mask;
}

__inline__ __device__ uint __pack_8_24( uint a, uint b )
{
    uint d;
    asm( "bfi.b32  %0, %1, %2, 24, 8;" : "=r"( d ) : "r"( a ), "r"( b ) );
    return d;
}

__inline__ __device__ uint2 __unpack_8_24( uint d )
{
    uint a;
    asm( "bfe.u32  %0, %1, 24, 8;" : "=r"( a ) : "r"( d ) );
    return make_uint2( a, d & 0x00FFFFFFU );
}

#include "dpd/dev/core.h"

#define MYCPBX  (4)
#define MYCPBY  (2)
#define MYCPBZ  (2)
#define MYWPB   (4)

#include "dpd/dev/merged.h"

__global__ void make_texture2( uint2 *start_and_count, const int *start, const int *count, const int n )
{
    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ) {
        start_and_count[i] = make_uint2( start[i], count[i] );
    }
}

#include "dpd/dev/transpose.h"
#include "dpd/imp/flocal.h"
