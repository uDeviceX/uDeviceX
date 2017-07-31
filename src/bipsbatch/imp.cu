#include "bipsbatch/imp.h"
#include "bipsbatch/dev.map.h"
#include "bipsbatch/dev.h"

namespace bipsbatch {

static void get_start(SFrag sfrag[26], /**/ unsigned int start[27]) {
    /* generate padded start */
    int i;
    start[0] = 0;
    for (i = 0; i < 26; ++i) start[i + 1] = start[i] + 16 * (((unsigned int)sfrag[i].n + 15) / 16);
}

void interactions(const SFrag ssfrag[26], const Frag ffrag[26], const Rnd rrnd[26], /**/ float *ff) {
    static unsigned int start[27];
    int n; /* number of threads */
    get_start(ssfrag, /**/ start);
    n = 2 * start[26];
    
    CC(cudaMemcpyToSymbolAsync(dev::start, start,   sizeof(start), 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::ssfrag, ssfrag, sizeof(SFrag) * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::ffrag, ffrag,   sizeof(Frag)  * 26, 0, H2D));
    CC(cudaMemcpyToSymbolAsync(dev::rrnd,   rrnd,   sizeof(Rnd)   * 26,  0, H2D));

    dSync();
    if (n) dev::force <<<k_cnf(n)>>> (ff);
}

};
