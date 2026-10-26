// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/mesh/bbox/imp.cpp */
#include <vector_types.h>

#ifndef UDX_EXCH_TYPE_COMMON
#define UDX_EXCH_TYPE_COMMON
/* begin src/inc/type.h */
template <typename T, int N>
struct Sarray { /* [s]tatic array */
    T d[N];
};

struct Particle {
    float r[3], v[3];
};

struct Rigid {
    float Iinv[6],            /* moment of inertia            6        */
        mass,                 /* mass of the solid            7        */
        com[3],               /* [c]enter [o]f [m]ass         10       */
        v[3], om[3],          /* linear and angular velocity  13 16    */
        e0[3], e1[3], e2[3],  /* local referential            19 22 25 */
        fo[3], to[3],         /* force, torque                28 31    */
        id;                   /* id of the solid              32       */
};

struct Force {
    float f[3];
};

struct Momentum {
    float P[3], L[3]; /* linear and angular momentum */
};

struct Positioncp {
    float rc[3]; /* current position  */
    float rp[3]; /* previous position */
};

typedef Sarray<int, 26>       int26;
typedef Sarray<int, 27>       int27;

typedef Sarray<int*, 26>     intp26;
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*,    26> Fop26;
typedef Sarray<Momentum*, 26> Mop26;
typedef Sarray<Positioncp*, 26> Pocpp26;

/* particles wrap */
struct PaWrap {
    int n;
    const Particle *pp;
};

/* forces wrap */
struct FoWrap {
    int n;
    Force *ff;
};
/* end src/inc/type.h */
#endif
/* begin src/algo/minmax/imp.h */
struct Particle;
struct float3;
// tag::interface[]
void minmax(const Particle *pp, int nv, int nobj, /**/ float3 *lo, float3 *hi);
// end::interface[]
/* end src/algo/minmax/imp.h */

/* begin src/mesh/bbox/imp.h */
struct float3;
struct Particle;

void mesh_get_bbox(const float *rr, const int n, /**/ float3 *minbb, float3 *maxbb);

void mesh_get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);
void mesh_get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);
/* end src/mesh/bbox/imp.h */

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
/* end src/mesh/bbox/imp.cpp */

#line 1 "src/mesh/collision/imp.cu"
