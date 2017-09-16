#include "frag/imp.h"

void frag_estimates(int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}
