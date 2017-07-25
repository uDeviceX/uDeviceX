#include <cstdio>
#include <common.h>
#include "common.cuda.h"

#include "scan/int.h"

namespace scan {
namespace dev {
#include "scan/dev.h"
}

static void scan0(const unsigned char *input, int size, /**/ uint *output, /*w*/ uint *tmp) {
    enum { THREADS = 128 } ;

    if (tmp == NULL)
        cudaMalloc(&tmp, sizeof(uint) * (64 * 64 * 64 / THREADS));

    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;

    dev::breduce< THREADS / 32 ><<<nblocks, THREADS>>>((uint4 *)input, tmp, size / 16);

    dev::bexscan< THREADS ><<<1, THREADS, nblocks*sizeof(uint)>>>(tmp, nblocks);

    dev::gexscan< THREADS / 32 ><<<nblocks, THREADS>>>((uint4 *)input, tmp, (uint4 *)output, size / 16);
}

void scan(const int *input, int size, /**/ int *output, /*w*/ Work *w) {
    dev::compress <<< k_cnf(size) >>> (size, (const int4*) input, /**/ (uchar4 *) w->compressed);

    scan0(w->compressed, size, /**/ (uint*) output, /*w*/ w->tmp);
}

void alloc_work(/**/ Work *w) {

}

void free_work(/**/ Work *w) {

}
}
