#include <stdlib.h>
#include <vector_types.h>

#include "rbc/adj/type.h"
#include "rbc/adj/imp.h"

#include "rbc/edg/imp.h"

#include "imp.h"

namespace anti {
static void ini0(int md, int nv, adj::Hst *adj, /**/ int *hst, /*w*/ int *hx, int *hy) {
    int valid, i;
    adj::Map m;
    
    edg::ini(md, nv, /**/ hx);
    for (i = 0; i < md*nv; i++) {
        valid = adj::hst(md, nv, i, adj, /**/ &m);
        if (!valid) continue;
    }
}

void ini(int md, int nv, adj::Hst *adj, /**/ int *anti) {
    int n;
    int *hst, *hx, *hy;
    n = md*nv;
    hx  = (int*)malloc(n*sizeof(int));
    hy  = (int*)malloc(n*sizeof(int));
    ini0(md, nv, adj, /**/ hst, /*w*/ hx, hy);
    free(hx); free(hy);
}

} /* namespace */

