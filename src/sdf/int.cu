#include <cstdio>
#include <conf.h>
#include "inc/conf.h"
#include "cc.h"

#include "inc/type.h"
#include "common.h"
#include "common.cuda.h"
#include "inc/macro.h"

#include "sdf/type.h"
#include "sdf/int.h"
#include "sdf/imp.h"

namespace sdf {
void alloc_quants(Quants *q) {
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    CC(cudaMalloc3DArray(&q->arrsdf, &fmt, make_cudaExtent(XTE, YTE, ZTE)));
}

void  free_quants(Quants *q) {
    CC(cudaFreeArray(q->arrsdf));
    q->texsdf.destroy();
}

void ini(Quants *q) {
    sub::ini(q->arrsdf, &q->texsdf);
}

void bulk_wall(const tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n) {
    sub::bulk_wall(texsdf, /*io*/ s_pp, s_n, /*o*/ w_pp, w_n);
}

int who_stays(const Quants q, Particle *pp, int n, int nc, int nv, int *stay) {
    return sub::who_stays(q.texsdf, pp, n, nc, nv, /**/ stay);
}

void bounce(const Quants *q, int n, /**/ Particle *pp) {
    sub::bounce(q->texsdf, n, /**/ pp);
}
}
