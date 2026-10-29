#define FRAG_HOST
#include "config.cuh"
#include "fragment.cuh"
#undef FRAG_HOST

namespace frag_hst {

void estimates(int3 L, int nfrags, float maxd, int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = ncell(L, i);
        cap[i] = (int)(e * maxd);
    }
}

}
