#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "common.h"
#include "inc/dev.h"
#include "msg.h"
#include "cc.h"
#include "kl.h"
#include "d/api.h"

#include "scan/int.h"
#include "scan/dev.h"

namespace scan {

enum { THREADS = 128 };

static void scan0(const unsigned char *input, int size, /**/ uint *output, /*w*/ uint *tmp) {
    int nblocks = ((size / 16) + THREADS - 1 ) / THREADS;
    KL(dev::breduce<THREADS/32>, (nblocks, THREADS                ), ((uint4 *)input, tmp, size / 16));
    KL(dev::bexscan<THREADS>   , (1, THREADS, nblocks*sizeof(uint)), (tmp, nblocks));
    KL(dev::gexscan<THREADS/32>, (nblocks, THREADS                ), ((uint4 *)input, tmp, (uint4 *)output, size / 16));
}

void scan(const int *input, int size, /**/ int *output, /*w*/ Work *w) {
    KL(dev::compress, (k_cnf(size)), (size, (const int4*) input, /**/ (uchar4 *) w->compressed));
    scan0(w->compressed, size, /**/ (uint*) output, /*w*/ w->tmp);
}

void alloc_work(int size, /**/ Work *w) {
    Dalloc0(&w->tmp, 64 * 64 * 64 / THREADS);
    Dalloc0(&w->compressed, 4 * size);
}

void free_work(/**/ Work *w) {
    Dfree0(w->tmp);
    Dfree0(w->compressed);
}

} /* namespace */
