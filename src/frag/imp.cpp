#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"
#include "frag/imp.h"

#define FRAG_HOST
#include "dev.h"
#undef FRAG_HOST

namespace fraghst {

void estimates(int3 L, int nfrags, float maxd, /**/ int *cap) {
    int i, e;
    for (i = 0; i < nfrags; ++i) {
        e = ncell(L, i);
        e = (int) (e * maxd);
        cap[i] = e;
    }
}

} // fraghst
