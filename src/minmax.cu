#include <mpi.h>
#include ".conf.h"
#include "m.h"
#include "common.h"
#include "minmax.h"

#define MAXTHREADS 1024
#define WARPSIZE     32
#define MAXV     100000000.
#define MINV    -100000000.

typedef struct
{
    int g_block_id;
    int g_blockcnt;
    float3 minval;
    float3 maxval;
} sblockds_t;

__global__ void minmaxob(const Particle * const d_data, float *bboxes, int size) {
    __shared__ float3 mintemp[32];
    __shared__ float3 maxtemp[32];
    __shared__ float shrtmp[3][MAXTHREADS];

    float3 mintemp1, maxtemp1;
    float3 mindef, maxdef;
    float temp2;
    mindef.x=MAXV;   mindef.y=MAXV;   mindef.z=MAXV;
    maxdef.x=MINV;   maxdef.y=MINV;   maxdef.z=MINV;
    __syncthreads();
    int tid = threadIdx.x;
    int xyz;
    for(int i=tid; i<3*blockDim.x; i+=blockDim.x) {
        xyz=i%3;
        shrtmp[xyz][i/3] = (i/3<size)?d_data[i/3+blockIdx.x*size].r[xyz]:MINV;
    }
    __syncthreads();
    mintemp1.x = (tid<size)?shrtmp[0][tid]:MAXV;
    mintemp1.y = (tid<size)?shrtmp[1][tid]:MAXV;
    mintemp1.z = (tid<size)?shrtmp[2][tid]:MAXV;
    maxtemp1.x = (tid<size)?shrtmp[0][tid]:MINV;
    maxtemp1.y = (tid<size)?shrtmp[1][tid]:MINV;
    maxtemp1.z = (tid<size)?shrtmp[2][tid]:MINV;
    for (int d=1; d<32; d<<=1) {
        temp2 = __shfl_up(mintemp1.x,d);
        mintemp1.x=(mintemp1.x>temp2)?temp2:mintemp1.x;
        temp2 = __shfl_up(mintemp1.y,d);
        mintemp1.y=(mintemp1.y>temp2)?temp2:mintemp1.y;
        temp2 = __shfl_up(mintemp1.z,d);
        mintemp1.z=(mintemp1.z>temp2)?temp2:mintemp1.z;
        temp2 = __shfl_up(maxtemp1.x,d);
        maxtemp1.x=(maxtemp1.x<temp2)?temp2:maxtemp1.x;
        temp2 = __shfl_up(maxtemp1.y,d);
        maxtemp1.y=(maxtemp1.y<temp2)?temp2:maxtemp1.y;
        temp2 = __shfl_up(maxtemp1.z,d);
        maxtemp1.z=(maxtemp1.z<temp2)?temp2:maxtemp1.z;
    }
    if (tid%32 == 31) {
        mintemp[tid/32] = mintemp1;
        maxtemp[tid/32] = maxtemp1;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        mintemp1= (tid < blockDim.x/32)?mintemp[threadIdx.x]:mindef;
        maxtemp1= (tid < blockDim.x/32)?maxtemp[threadIdx.x]:maxdef;
        for (int d=1; d<32; d<<=1) {
            temp2 = __shfl_up(mintemp1.x,d);
            mintemp1.x=(mintemp1.x>temp2)?temp2:mintemp1.x;
            temp2 = __shfl_up(mintemp1.y,d);
            mintemp1.y=(mintemp1.y>temp2)?temp2:mintemp1.y;
            temp2 = __shfl_up(mintemp1.z,d);
            mintemp1.z=(mintemp1.z>temp2)?temp2:mintemp1.z;
            temp2 = __shfl_up(maxtemp1.x,d);
            maxtemp1.x=(maxtemp1.x<temp2)?temp2:maxtemp1.x;
            temp2 = __shfl_up(maxtemp1.y,d);
            maxtemp1.y=(maxtemp1.y<temp2)?temp2:maxtemp1.y;
            temp2 = __shfl_up(maxtemp1.z,d);
            maxtemp1.z=(maxtemp1.z<temp2)?temp2:maxtemp1.z;
        }
        if (tid < blockDim.x/32) {
            mintemp[tid] = mintemp1;
            maxtemp[tid] = maxtemp1;
        }
    }
    __syncthreads();
    if (threadIdx.x==blockDim.x-1) {
        const float3 mins = mintemp[blockDim.x/32-1];
        const float3 maxs = maxtemp[blockDim.x/32-1];
        bboxes [blockIdx.x * 6 + 0] = mins.x;
        bboxes [blockIdx.x * 6 + 1] = maxs.x;
        bboxes [blockIdx.x * 6 + 2] = mins.y;
        bboxes [blockIdx.x * 6 + 3] = maxs.y;
        bboxes [blockIdx.x * 6 + 4] = mins.z;
        bboxes [blockIdx.x * 6 + 5] = maxs.z;
    }

}


__global__ void minmaxmba(const Particle  *d_data, float *bboxes,
        int size, sblockds_t *ptoblockds) {

    __shared__ float3 mintemp[32];
    __shared__ float3 maxtemp[32];
    __shared__ float shrtmp[3][MAXTHREADS];

    __shared__ unsigned int my_blockId;
    const int which=blockIdx.x/((size+blockDim.x-1)/blockDim.x); /* which particle should manage */
    float3 mintemp1, maxtemp1;
    float3 mindef, maxdef;
    float temp2;
    if (threadIdx.x==0) {
        my_blockId = atomicAdd( &(ptoblockds[which].g_block_id), 1 );
    }
    mindef.x=MAXV;   mindef.y=MAXV;   mindef.z=MAXV;
    maxdef.x=MINV;   maxdef.y=MINV;   maxdef.z=MINV;
    __syncthreads();
    int tid = threadIdx.x;
    int xyz;
    for(int i=tid; i<3*blockDim.x; i+=blockDim.x) {
        xyz=i%3;
        shrtmp[xyz][i/3] = (i/3+my_blockId*blockDim.x<size)?d_data[i/3+my_blockId*blockDim.x+which*size].r[xyz]:MINV;
    }
    __syncthreads();
    mintemp1.x = (tid+my_blockId*blockDim.x<size)?shrtmp[0][tid]:MAXV;
    mintemp1.y = (tid+my_blockId*blockDim.x<size)?shrtmp[1][tid]:MAXV;
    mintemp1.z = (tid+my_blockId*blockDim.x<size)?shrtmp[2][tid]:MAXV;
    maxtemp1.x = (tid+my_blockId*blockDim.x<size)?shrtmp[0][tid]:MINV;
    maxtemp1.y = (tid+my_blockId*blockDim.x<size)?shrtmp[1][tid]:MINV;
    maxtemp1.z = (tid+my_blockId*blockDim.x<size)?shrtmp[2][tid]:MINV;
    for (int d=1; d<32; d<<=1) {
        temp2 = __shfl_up(mintemp1.x,d);
        mintemp1.x=(mintemp1.x>temp2)?temp2:mintemp1.x;
        temp2 = __shfl_up(mintemp1.y,d);
        mintemp1.y=(mintemp1.y>temp2)?temp2:mintemp1.y;
        temp2 = __shfl_up(mintemp1.z,d);
        mintemp1.z=(mintemp1.z>temp2)?temp2:mintemp1.z;
        temp2 = __shfl_up(maxtemp1.x,d);
        maxtemp1.x=(maxtemp1.x<temp2)?temp2:maxtemp1.x;
        temp2 = __shfl_up(maxtemp1.y,d);
        maxtemp1.y=(maxtemp1.y<temp2)?temp2:maxtemp1.y;
        temp2 = __shfl_up(maxtemp1.z,d);
        maxtemp1.z=(maxtemp1.z<temp2)?temp2:maxtemp1.z;
    }
    if (tid%32 == 31) {
        mintemp[tid/32] = mintemp1;
        maxtemp[tid/32] = maxtemp1;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        mintemp1= (tid < blockDim.x/32)?mintemp[threadIdx.x]:mindef;
        maxtemp1= (tid < blockDim.x/32)?maxtemp[threadIdx.x]:maxdef;
        for (int d=1; d<32; d<<=1) {
            temp2 = __shfl_up(mintemp1.x,d);
            mintemp1.x=(mintemp1.x>temp2)?temp2:mintemp1.x;
            temp2 = __shfl_up(mintemp1.y,d);
            mintemp1.y=(mintemp1.y>temp2)?temp2:mintemp1.y;
            temp2 = __shfl_up(mintemp1.z,d);
            mintemp1.z=(mintemp1.z>temp2)?temp2:mintemp1.z;
            temp2 = __shfl_up(maxtemp1.x,d);
            maxtemp1.x=(maxtemp1.x<temp2)?temp2:maxtemp1.x;
            temp2 = __shfl_up(maxtemp1.y,d);
            maxtemp1.y=(maxtemp1.y<temp2)?temp2:maxtemp1.y;
            temp2 = __shfl_up(maxtemp1.z,d);
            maxtemp1.z=(maxtemp1.z<temp2)?temp2:maxtemp1.z;
        }
        if (tid < blockDim.x/32) {
            mintemp[tid] = mintemp1;
            maxtemp[tid] = maxtemp1;
        }
    }
    __syncthreads();
    mintemp1=mintemp[blockDim.x/32-1];
    maxtemp1=maxtemp[blockDim.x/32-1];
    if (threadIdx.x==(blockDim.x-1)) {
        do {} while( atomicAdd(&(ptoblockds[which].g_blockcnt),0) < my_blockId );
        mintemp1.x=(ptoblockds[which].minval.x<mintemp1.x)?ptoblockds[which].minval.x:mintemp1.x;
        maxtemp1.x=(ptoblockds[which].maxval.x>maxtemp1.x)?ptoblockds[which].maxval.x:maxtemp1.x;
        mintemp1.y=(ptoblockds[which].minval.y<mintemp1.y)?ptoblockds[which].minval.y:mintemp1.y;
        maxtemp1.y=(ptoblockds[which].maxval.y>maxtemp1.y)?ptoblockds[which].maxval.y:maxtemp1.y;
        mintemp1.z=(ptoblockds[which].minval.z<mintemp1.z)?ptoblockds[which].minval.z:mintemp1.z;
        maxtemp1.z=(ptoblockds[which].maxval.z>maxtemp1.z)?ptoblockds[which].maxval.z:maxtemp1.z;
        if(my_blockId==(((size+blockDim.x-1)/blockDim.x))-1) { /* it is the last block; reset for next iteration */
            ptoblockds[which].minval=mindef;
            ptoblockds[which].maxval=maxdef;
            ptoblockds[which].g_blockcnt=0;
            ptoblockds[which].g_block_id=0;
            bboxes[6*which + 0] = mintemp1.x;
            bboxes[6*which + 1] = maxtemp1.x;
            bboxes[6*which + 2] = mintemp1.y;
            bboxes[6*which + 3] = maxtemp1.y;
            bboxes[6*which + 4] = mintemp1.z;
            bboxes[6*which + 5] = maxtemp1.z;
        } else {
            ptoblockds[which].minval=mintemp1;
            ptoblockds[which].maxval=maxtemp1;
            atomicAdd(&(ptoblockds[which].g_blockcnt),1);
        }
    }

}

void minmax(const Particle * const rbc, int size, int n, float *bboxes)
{
    const int size32 = ((size + 31) / 32) * 32;

    if (size32 < MAXTHREADS)
        minmaxob<<<n, size32>>>(rbc, bboxes, size);
    else
    {
        static int nctc = -1;

        static sblockds_t *ptoblockds = NULL;

        if( n > nctc)
        {
            sblockds_t * h_ptoblockds = new sblockds_t[n];

            for(int i=0; i < n; i++)
            {
                h_ptoblockds[i].g_block_id=0;
                h_ptoblockds[i].g_blockcnt=0;
                h_ptoblockds[i].minval.x=MAXV;
                h_ptoblockds[i].maxval.x=MINV;
                h_ptoblockds[i].minval.y=MAXV;
                h_ptoblockds[i].maxval.y=MINV;
                h_ptoblockds[i].minval.z=MAXV;
                h_ptoblockds[i].maxval.z=MINV;
            }

            if (ptoblockds != NULL)
                CC(cudaFree(ptoblockds));

            CC(cudaMalloc((void **)&ptoblockds,sizeof(sblockds_t) * n));

            CC(cudaMemcpy(ptoblockds, h_ptoblockds, sizeof(sblockds_t) * n, H2D));

            delete [] h_ptoblockds;
        }

        int nblocks= n * ((size + MAXTHREADS - 1) / MAXTHREADS);

        minmaxmba<<<nblocks, MAXTHREADS>>>(rbc, bboxes, size, ptoblockds);
    }
}
