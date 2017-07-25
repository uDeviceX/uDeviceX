#include <cstdio>
#include <common.h>
#include "common.cuda.h"

#include "scan/int.h"

namespace dev {
#include "scan/dev.h"
}

static void scan0(const unsigned char *input, int size, uint *output) {
    enum { THREADS = 128 } ;

    static uint *tmp = NULL;

    if (tmp == NULL)
        cudaMalloc(&tmp, sizeof(uint) * (64 * 64 * 64 / THREADS));

    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;

    dev::breduce< THREADS / 32 ><<<nblocks, THREADS>>>((uint4 *)input, tmp, size / 16);

    dev::bexscan< THREADS ><<<1, THREADS, nblocks*sizeof(uint)>>>(tmp, nblocks);

    dev::gexscan< THREADS / 32 ><<<nblocks, THREADS>>>((uint4 *)input, tmp, (uint4 *)output, size / 16);
}

void scan(const int *input, int size, /**/ int *output, /*w*/ unsigned char *compressed) {
    dev::compress <<< k_cnf(size) >>> (size, (const int4*) input, /**/ (uchar4 *) compressed);

    scan0(compressed, size, /**/ (uint*) output);
}
