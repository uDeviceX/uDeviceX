#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"
#include "frag/imp.h"

#define FRAG_HOST
#include "dev.h"
#undef FRAG_HOST

namespace fraghst {

void frag_estimates(int nfrags, float maxd, /**/ int *cap) {
    int3 L;
    L.x = XS; L.y = YS; L.z = ZS;
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = frag_ncell(L, i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

} // fraghst
