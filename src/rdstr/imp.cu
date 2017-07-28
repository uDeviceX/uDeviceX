#include "common.h"

#include "minmax.h"
#include "rdstr/imp.h"

namespace rdstr {
namespace sub {

enum {X, Y, Z};

void extents(const Particle *pp, int nc, int nv, /**/ float3 *ll, float3 *hh) {
    if (nc) minmax(pp, nv, nc, /**/ ll, hh);
}

void get_pos(int n, const float3 *ll, const float3 *hh, /**/ float *rr) {
    for (int i = 0; i < n; ++i) {
        float3 l = ll[i], h = hh[i];
        float *r = rr + 3 * i;
        r[X] = 0.5f * (l.x + h.x);
        r[Y] = 0.5f * (l.y + h.y);
        r[Z] = 0.5f * (l.z + h.z);
    }
}



} // sub
} // rdstr
