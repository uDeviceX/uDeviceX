#include "inc/type.h"
#include "algo/minmax/imp.h"

#include "mesh/bbox.h"

static void T2r(const int i, const Particle *pp, const float **r) {*r = pp[i].r;}
static void T2r(const int i, const float *rr,    const float **r) {*r = rr + 3*i;}

template <typename T>
static void get_bbox_(const T *vv, const int n, /**/ float3 *minbb, float3 *maxbb) {
    if (n == 0) return;

    const float *r;
    T2r(0, vv, /**/ &r);

    float3 minb = make_float3(r[0], r[1], r[2]);
    float3 maxb = make_float3(r[0], r[1], r[2]);

    for (int i = 1; i < n; ++i) {
        T2r(i, vv, /**/ &r);
        minb.x = min(minb.x, r[0]); maxb.x = max(maxb.x, r[0]);
        minb.y = min(minb.y, r[1]); maxb.y = max(maxb.y, r[1]);
        minb.z = min(minb.z, r[2]); maxb.z = max(maxb.z, r[2]);
    }
    *minbb = minb; *maxbb = maxb;
}

void mesh_get_bbox(const float *rr, const int n, /**/ float3 *minbb, float3 *maxbb) {
    get_bbox_(rr, n, /**/ minbb, maxbb);
}

void mesh_get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb) {
    for (int i = 0; i < ns; ++i)
    get_bbox_(pp + i*nps, nps, /**/ minbb + i, maxbb + i);
}

void mesh_get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb) {
    if (ns == 0) return;
    minmax(pp, nps, ns, /**/ minbb, maxbb);
}
