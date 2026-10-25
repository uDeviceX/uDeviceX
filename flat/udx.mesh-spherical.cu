// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/mesh/spherical/imp.cpp */
#include <stdio.h>
#include <math.h>

/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */
/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

/* begin src/algo/vectors/imp.h */
struct Particle;
struct Vectors;
struct Coords;

// tag::interface[]
void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>
// end::interface[]
/* end src/algo/vectors/imp.h */
/* begin src/algo/kahan_sum/imp.h */
struct KahanSum;

// tag::interface[]
void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>
// end::interface[]
/* end src/algo/kahan_sum/imp.h */

/* begin src/mesh/spherical/imp.h */
struct MeshSpherical;
struct Vectors;

// tag::interface[]
void mesh_spherical_ini(int nv, /**/ MeshSpherical**);
void mesh_spherical_fin(MeshSpherical*);
void  mesh_spherical_apply(MeshSpherical*, int nm, Vectors*, /**/
                           double *r, double *theta, double *phi);
// end::interface[]
/* end src/mesh/spherical/imp.h */
/* begin src/mesh/spherical/imp/util.h */
static void compute_com(int nv, Vectors *pos, /**/ double *com) {
    enum {X, Y, Z};
    int i;
    float r[3];
    KahanSum *sx, *sy, *sz;
    kahan_sum_ini(&sx); kahan_sum_ini(&sy); kahan_sum_ini(&sz);
    for (i = 0; i < nv; i++) {
        UC(vectors_get(pos, i, /**/ r));
        kahan_sum_add(sx, r[X]);
        kahan_sum_add(sy, r[Y]);
        kahan_sum_add(sz, r[Z]);
    }
    com[X] = kahan_sum_get(sx)/nv;
    com[Y] = kahan_sum_get(sy)/nv;
    com[Z] = kahan_sum_get(sz)/nv;
    kahan_sum_fin(sx); kahan_sum_fin(sy); kahan_sum_fin(sz);
}

static void to_com(int nv, int offset, Vectors *pos, /**/ double *rr) {
    enum {X, Y, Z};
    int i;
    float   r0[3]; /* from */
    double *r1;    /* to */
    double com[3];
    UC(compute_com(nv, pos, /**/ com));
    for (i = 0; i < nv; i++) {
        UC(vectors_get(pos, i + offset, /**/ r0));
        r1 = &rr[3*i];
        r1[X] = r0[X] - com[X];
        r1[Y] = r0[Y] - com[Y];
        r1[Z] = r0[Z] - com[Z];
    }
}
/* end src/mesh/spherical/imp/util.h */
/* begin src/mesh/spherical/imp/type.h */
struct MeshSpherical {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int nv;
};
/* end src/mesh/spherical/imp/type.h */
/* begin src/mesh/spherical/imp/main.h */
void mesh_spherical_ini(int nv, MeshSpherical **pq) {
    MeshSpherical *q;
    EMALLOC(1, &q);
    EMALLOC(3*nv, &q->rr);
    q->nv = nv;
    *pq = q;
}

void mesh_spherical_fin(MeshSpherical *q) {
    EFREE(q->rr);
    EFREE(q);
}

static void xyz2sph(double x, double y, double z, /**/ double *pr, double *ptheta, double *pphi) {
    double r, theta, phi;
    r = sqrt(x*x + y*y + z*z);
    theta = atan2(z, r);
    phi = atan2(y, x);
    *pr = r; *ptheta = theta; *pphi = phi;
}
static void apply(int n, double *rr, /**/ double *r, double *theta, double *phi) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < n; i++) {
        xyz2sph(rr[X], rr[Y], rr[Z], /**/ r, theta, phi);
        rr += 3; r++; theta++; phi++;
    }
}
void mesh_spherical_apply(MeshSpherical *q, int m, Vectors *pos, /**/ double *r, double *theta, double *phi) {
    int i, nv, offset;
    double *rr;
    nv = q->nv; rr = q->rr; offset = 0;
    for (i = 0; i < m; i++) {
        UC(to_com(nv, offset, pos, /**/ rr));
        apply(nv, rr, /**/ r, theta, phi);
        rr += 3*nv; r += nv; theta += nv; phi += nv; offset += nv;
    }
}
/* end src/mesh/spherical/imp/main.h */
/* end src/mesh/spherical/imp.cpp */

#line 1 "src/mesh/tri_area/imp.cpp"
