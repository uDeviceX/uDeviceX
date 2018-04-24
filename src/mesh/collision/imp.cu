#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/def.h"
#include "utils/error.h"
#include "utils/msg.h"
#include "utils/cc.h"

#include "d/q.h"
#include "d/ker.h"
#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/texo.h"
#include "utils/texo.dev.h"
#include "mesh/triangles/type.h"
#include "mesh/triangles/imp.h"

#include "utils/kl.h"
#include "rigid/imp.h"

#include "imp.h"

enum {X, Y, Z};

static __host__ __device__ bool same_side(const float *x, const float *p, const float *a, const float *b, const float *inplane) {
    const float n[3] = {a[Y] * b[Z] - a[Z] * b[Y],
                        a[Z] * b[X] - a[X] * b[Z],
                        a[X] * b[Y] - a[Y] * b[X]};

    const float ndx = n[X] * (x[X] - inplane[X]) + n[Y] * (x[Y] - inplane[Y]) + n[Z] * (x[Z] - inplane[Z]);
    const float ndp = n[X] * (p[X] - inplane[X]) + n[Y] * (p[Y] - inplane[Y]) + n[Z] * (p[Z] - inplane[Z]);

    return ndx * ndp > 0;
}

static __host__ __device__ bool in_tetrahedron(const float *x, const float *A, const float *B, const float *C, const float *D) {
    const float AB[3] = {B[X] - A[X], B[Y] - A[Y], B[Z] - A[Z]};
    const float AC[3] = {C[X] - A[X], C[Y] - A[Y], C[Z] - A[Z]};
    const float AD[3] = {D[X] - A[X], D[Y] - A[Y], D[Z] - A[Z]};

    const float BC[3] = {C[X] - B[X], C[Y] - B[Y], C[Z] - B[Z]};
    const float BD[3] = {D[X] - B[X], D[Y] - B[Y], D[Z] - B[Z]};

    return
        same_side(x, A, BC, BD, D) &&
        same_side(x, B, AD, AC, D) &&
        same_side(x, C, AB, BD, D) &&
        same_side(x, D, AB, AC, A);
}

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt) {
    int c = 0;
    float origin[3] = {0, 0, 0};

    if (spdir != NOT_PERIODIC)
        origin[spdir] = r[spdir];

    for (int i = 0; i < nt; ++i) {
        int4 t = tt[i];
        if (in_tetrahedron(r, vv + 3*t.x, vv + 3*t.y, vv + 3*t.z, origin)) ++c;
    }
    return c%2;
}

namespace collision_dev
{
__global__ void init_tags(const int n, const int color, /**/ int *tags) {
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < n) tags[gid] = color;
}

union Pos {
    float2 f2[2];
    struct { float r[3]; float dummy; };
};

static __device__ Pos tex2Pos(const Texo<float2> texvert, const int id) {
    Pos r;
    r.f2[0] = texo_fetch(texvert, 3 * id + 0);
    r.f2[1] = texo_fetch(texvert, 3 * id + 1);
    return r;
}

static __device__ bool inside_box(const float r[3], float3 lo, float3 hi) {
    enum {X, Y, Z};
    return
        r[X] >= lo.x && r[X] <= hi.x &&
        r[Y] >= lo.y && r[Y] <= hi.y &&
        r[Z] >= lo.z && r[Z] <= hi.z;
}

/* assume nm blocks along y */
__global__ void label_tex(int pdir, const Particle *pp, const int n, const Texo<float2> texvert, const int nv,
                          Triangles tri, const float3 *minext, const float3 *maxext,
                          int lab_in, /**/ int *labels) {
    int i, sid, gid, count, mbase;
    Particle p;
    Pos a, b, c;
    float3 lo, hi;
    int4 t;
    sid = blockIdx.y;
    gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n) return;

    count = 0;

    p = pp[gid];
    
    lo = minext[sid];
    hi = maxext[sid];
    if (!inside_box(p.r, lo, hi)) return;

    float origin[3] = {0, 0, 0};
    if (pdir != NOT_PERIODIC) origin[pdir] = p.r[pdir];

    mbase = nv * sid;
    for (i = 0; i < tri.nt; ++i) {
        t = tri.tt[i];

        a = tex2Pos(texvert, mbase + t.x);
        b = tex2Pos(texvert, mbase + t.y);
        c = tex2Pos(texvert, mbase + t.z);

        if (in_tetrahedron(p.r, a.r, b.r, c.r, origin)) ++count;
    }

    // dont consider the case of inside several solids
    if (count % 2) atomicExch(labels + gid, lab_in);
}
}

static void label(int pdir, int n, const Particle *pp, const Triangles *tri, int nv, int nm, const Texo<float2> texvert,                        
                       const float3 *minext, const float3 *maxext, int lab_in, int lab_out, /**/ int *labels) {
    enum {X, Y, Z};
    if (nm == 0 || n == 0) return;

    KL(collision_dev::init_tags, (k_cnf(n)), (n, lab_out, /**/ labels));

    enum {THR = 128};
    dim3 thrd(THR, 1);
    dim3 blck(ceiln(n, THR), nm);

    KL(collision_dev::label_tex, (blck, thrd),
       (pdir, pp, n, texvert, nv, *tri, minext, maxext, lab_in, /**/ labels)); 
}

void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri, 
                     int nv, int nm, const Particle *i_pp, 
                     const float3 *minext, const float3 *maxext,
                     int lab_in, int lab_out, /**/ int *labels) {
    Texo<float2> texvert;
    int ntex;
    ntex = 3 * nm * nv;
    
    if (nm == 0 || n == 0) return;
    texo_setup(ntex, (float2*) i_pp, /**/ &texvert);
    UC(label(pdir, n, pp, tri, nv, nm, texvert, minext, maxext, lab_in, lab_out, /**/ labels));
    texo_destroy(&texvert);
}
