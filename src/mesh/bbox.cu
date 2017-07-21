#include "common.h"
#include "minmax.h"

#include "mesh/bbox.h"

namespace mesh {
static void get_bbox(const Particle *pp, const int n, /**/ float3 *minbb, float3 *maxbb) {
    if (n == 0) return;

    const float *r = pp[0].r;

    float3 minb = make_float3(r[0], r[1], r[2]);
    float3 maxb = make_float3(r[0], r[1], r[2]);

    auto min = [](float a, float b) {return a > b ? b : a;};
    auto max = [](float a, float b) {return a > b ? a : b;};
    
    for (int i = 1; i < n; ++i) {
        r = pp[i].r;
        minb.x = min(minb.x, r[0]); maxb.x = max(maxb.x, r[0]);
        minb.y = min(minb.y, r[1]); maxb.y = max(maxb.y, r[1]);
        minb.z = min(minb.z, r[2]); maxb.z = max(maxb.z, r[2]);
    }
    *minbb = minb; *maxbb = maxb;
}

void get_bbox(const float *rr, const int n, /**/ float3 *minbb, float3 *maxbb) {
    if (n == 0) return;

    const float *r = rr;
        
    float3 minb = make_float3(r[0], r[1], r[2]);
    float3 maxb = make_float3(r[0], r[1], r[2]);

    auto min = [](float a, float b) {return a > b ? b : a;};
    auto max = [](float a, float b) {return a > b ? a : b;};

    for (int i = 1; i < n; ++i) {
        r = rr + 3 * i;
        minb.x = min(minb.x, r[0]); maxb.x = max(maxb.x, r[0]);
        minb.y = min(minb.y, r[1]); maxb.y = max(maxb.y, r[1]);
        minb.z = min(minb.z, r[2]); maxb.z = max(maxb.z, r[2]);
    }
    *minbb = minbb; *maxbb = maxbb;
}

void get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb) {
    for (int i = 0; i < ns; ++i)
    get_bbox(pp + i*nps, nps, /**/ minbb + i, maxbb + i);
}

void get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb) {
    if (ns == 0) return;
    minmax(pp, nps, ns, /**/ minbb, maxbb);
}
}
