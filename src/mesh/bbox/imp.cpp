#include <vector_types.h>

#include "inc/type.h"
#include "algo/minmax/imp.h"

#include "imp.h"

static void T2r(const int i, const Particle *pp, const float **r) {*r = pp[i].r;}
static void T2r(const int i, const float *rr,    const float **r) {*r = rr + 3*i;}

static float min2(float a, float b) {return a < b ? a : b;}
static float max2(float a, float b) {return a < b ? b : a;}

template <typename T>
static void get_bbox_(const T *vv, const int n, /**/ float3 *minbb, float3 *maxbb) {
    enum {X, Y, Z};
    const float *r;
    float3 minb, maxb;
    
    if (n == 0) return;

    T2r(0, vv, /**/ &r);

    minb.x = r[X]; minb.y = r[Y]; minb.z = r[Z];
    maxb = minb;

    for (int i = 1; i < n; ++i) {
        T2r(i, vv, /**/ &r);
        minb.x = min2(minb.x, r[0]); maxb.x = max2(maxb.x, r[0]);
        minb.y = min2(minb.y, r[1]); maxb.y = max2(maxb.y, r[1]);
        minb.z = min2(minb.z, r[2]); maxb.z = max2(maxb.z, r[2]);
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
