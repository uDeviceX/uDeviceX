#include <mpi.h>
#include <stdio.h>
#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "inc/def.h"
#include "msg.h"
#include "utils/cc.h"
#include "utils/error.h"
#include "utils/halloc.h"

#include "d/q.h"
#include "d/ker.h"
#include "d/api.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "inc/macro.h"

#include "utils/kl.h"
#include "glb/get.h"
#include "inc/dev/wvel.h"

#include "sdf/field/imp.h"

#include "sdf/type.h"
#include "imp.h"
#include "dev/cheap.h"
#include "dev/main.h"

namespace sdf {
namespace sub {

struct Tex { /* simplifies communication between ini[0123..] */
    cudaArray *a;
    tex3Dca<float> *t;
};

static void ini0(float *D, /**/ struct Tex te) {
    cudaMemcpy3DParms copyParams;
    memset(&copyParams, 0, sizeof(copyParams));
    copyParams.srcPtr = make_cudaPitchedPtr((void*)D, XTE * sizeof(float), XTE, YTE);
    copyParams.dstArray = te.a;
    copyParams.extent = make_cudaExtent(XTE, YTE, ZTE);
    copyParams.kind = H2D;
    CC(cudaMemcpy3D(&copyParams));
    te.t->setup(te.a);
}

static void ini1(int N[3], float *D0, float *D1, /**/ struct Tex te) {
    int c;
    int L[3] = {XS, YS, ZS};
    int M[3] = {XWM, YWM, ZWM}; /* margin and texture */
    int T[3] = {XTE, YTE, ZTE};
    float G; /* domain size ([g]lobal) */
    float lo; /* left edge of subdomain */
    float org[3], spa[3]; /* origin and spacing */
    for (c = 0; c < 3; ++c) {
        G = m::dims[c] * L[c];
        lo = m::coords[c] * L[c];
        spa[c] = N[c] * (L[c] + 2 * M[c]) / G / T[c];
        org[c] = N[c] * (lo - M[c]) / G;
    }
    UC(field::sample(org, spa, N, D0,   T, /**/ D1));
    UC(ini0(D1, te));
}

static void ini2(int N[3], float* D0, /**/ struct Tex te) {
    float *D1 = new float[XTE * YTE * ZTE];
    UC(ini1(N, D0, D1, /**/ te));
    delete[] D1;
}

static void ini3(MPI_Comm cart, int N[3], float ext[3], float* D, /**/ struct Tex te) {
    enum {X, Y, Z};
    float sc, G; /* domain size in x ([G]lobal) */
    G = m::dims[X] * XS;
    sc = G / ext[X];
    UC(field::scale(N, sc, /**/ D));

    if (field_dumps) UC(field::dump(cart, N, D));

    UC(ini2(N, D, /**/ te));
}

void ini(MPI_Comm cart, cudaArray *arrsdf, tex3Dca<float> *texsdf) {
    enum {X, Y, Z};
    float *D;     /* data */
    int N[3];     /* size of D */
    float ext[3]; /* extent */
    int n;
    char f[] = "sdf.dat";
    struct Tex te {arrsdf, texsdf};

    UC(field::ini_dims(f, /**/ N, ext));
    n = N[X] * N[Y] * N[Z];
    D = new float[n];
    UC(field::ini_data(f, n, /**/ D));
    UC(ini3(cart, N, ext, D, /**/ te));
    delete[] D;
}

/* sort solvent particle into remaining in solvent and turning into wall according to keys (all on hst) */
static void split_wall_solvent(const int *keys, /*io*/ int *s_n, Particle *s_pp, /**/ int *w_n, Particle *w_pp) {
    int n = *s_n;
    Particle p;
    int k, ia = 0, is = 0, iw = 0; /* all, solvent, wall */

    for (ia = 0; ia < n; ++ia) {
        k = keys[ia];
        p = s_pp[ia];
        
        if      (k == W_BULK) s_pp[is++] = p;
        else if (k == W_WALL) w_pp[iw++] = p;
    }
    *s_n = is;
    *w_n = iw;
}

/* sort solvent particle (dev) into remaining in solvent (dev) and turning into wall (hst)*/
static void bulk_wall0(const tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int* s_n,
                       /*o*/ Particle *w_pp, int *w_n, /*w*/ int *keys) {
    int n = *s_n, *keys_hst;
    Particle *s_pp_hst;
    UC(emalloc(n * sizeof(Particle), (void**) &s_pp_hst));
    UC(emalloc(n * sizeof(int), (void**) &keys_hst));
    
    KL(dev::fill,(k_cnf(n)), (texsdf, s_pp, n, keys));
    cD2H(keys_hst, keys, n);
    cD2H(s_pp_hst, s_pp, n);

    UC(split_wall_solvent(keys_hst, /*io*/ s_n, s_pp_hst, /**/ w_n, w_pp));
    cH2D(s_pp, s_pp_hst, *s_n);
                       
    free(s_pp_hst);
    free(keys_hst);
}

void bulk_wall(const tex3Dca<float> texsdf, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n) {
    int *keys;
    Dalloc(&keys, MAX_PART_NUM);
    UC(bulk_wall0(texsdf, s_pp, s_n, w_pp, w_n, keys));
    CC(d::Free(keys));
}

/* bulk predicate : is in bulk? */
static bool bulkp(int *keys, int i) { return keys[i] == W_BULK; }
static int who_stays0(int *keys, int nc, int nv, /*o*/ int *stay) {
    int c, v;  /* cell and vertex */
    int s = 0; /* how many stays? */
    for (c = 0; c < nc; ++c) {
        v = 0;
        while (v  < nv && bulkp(keys, v + nv * c)) v++;
        if    (v == nv) stay[s++] = c;
    }
    return s;
}

static int who_stays1(int *keys, int n, int nc, int nv, /*o*/ int *stay) {
    int nc0, *keys_hst;
    UC(emalloc(n*sizeof(int), (void**) &keys_hst));
    cD2H(keys_hst, keys, n);
    nc0 = who_stays0(keys_hst, nc, nv, /**/ stay);
    free(keys_hst);
    return nc0;
}

int who_stays(const tex3Dca<float> texsdf, Particle *pp, int n, int nc, int nv, /**/ int *stay) {
    int *keys;
    CC(d::Malloc((void **) &keys, n*sizeof(keys[0])));
    KL(dev::fill, (k_cnf(n)), (texsdf, pp, n, keys));
    nc = who_stays1(keys, n, nc, nv, /**/ stay);
    CC(d::Free(keys));
    return nc;
}

void bounce(const tex3Dca<float> texsdf, int n, /**/ Particle *pp) {
    KL(dev::bounce, (k_cnf(n)), (texsdf, n, /**/ (float2*) pp));
}

} // sub
} // sdf
