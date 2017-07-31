#include <conf.h>
#include <cstdio>

#include "common.h"
#include "common.cuda.h"
#include "forces.h"
#include "k/read.h"
#include "k/common.h"

#include <limits> /* for rnd */
#include <stdint.h>
#include "rnd.h"

#include "bipsbatch/imp.h"
#include "bipsbatch/dev.map.h"
#include "bipsbatch/dev.h"

namespace bipsbatch {

static void get_start(const SFrag sfrag[26], /**/ unsigned int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * (((unsigned int)sfrag[i].n + 15) / 16);
}

void interactions(const SFrag26 ssfrag, const Frag26 ffrag, const Rnd26 rrnd, /**/ float *ff) {
    int27 start;
    int n; /* number of threads */
    get_start(ssfrag.d, /**/ start.d);
    n = 2 * start.d[26];
    
    CC(cudaMemcpyToSymbolAsync(dev::ssfrag, ssfrag.d, sizeof(SFrag) * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::ffrag, ffrag.d,   sizeof(Frag)  * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::rrnd,   rrnd.d,   sizeof(Rnd)   * 26,  0, H2D));
    
    if (n) dev::force <<<k_cnf(n)>>> (start, /**/ ff);
}

};
