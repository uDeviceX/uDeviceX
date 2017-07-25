namespace dev {
#include "dev.h"
}

void scan(unsigned char *input, int size, uint *output) {
    enum { THREADS = 128 } ;

    static uint *tmp = NULL;

    if (tmp == NULL)
        cudaMalloc(&tmp, sizeof(uint) * (64 * 64 * 64 / THREADS));

    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;

    breduce< THREADS / 32 ><<<nblocks, THREADS>>>((uint4 *)input, tmp, size / 16);

    bexscan< THREADS ><<<1, THREADS, nblocks*sizeof(uint)>>>(tmp, nblocks);

    gexscan< THREADS / 32 ><<<nblocks, THREADS>>>((uint4 *)input, tmp, (uint4 *)output, size / 16);
}
