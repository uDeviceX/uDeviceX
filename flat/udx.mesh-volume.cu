// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/mesh/volume/imp.cpp */
#include <stdio.h>
#include <vector_types.h>

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

/* begin src/io/mesh_read/imp.h */
struct int4;
struct MeshRead;

// tag::interface[]
void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
// end::interface[]
/* end src/io/mesh_read/imp.h */
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

/* begin src/mesh/volume/imp.h */
struct MeshVolume;
struct Vectors;
struct MeshRead;

// tag::interface[]
void mesh_volume_ini(MeshRead*, /**/ MeshVolume**); // <1>
void mesh_volume_fin(MeshVolume*);

double mesh_volume_apply0(MeshVolume*, Vectors*);                      // <2>
void  mesh_volume_apply(MeshVolume*, int nm, Vectors*, /**/ double*); // <3>
// end::interface[]
/* end src/mesh/volume/imp.h */
/* begin src/mesh/volume/imp/type.h */
struct MeshVolume {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int4 *tt;
    int nv, nt;
};
/* end src/mesh/volume/imp/type.h */
/* begin src/mesh/volume/imp/main.h */
void mesh_volume_ini(MeshRead *mesh, MeshVolume **pq) {
    int nv, nt;
    MeshVolume *q;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    EMALLOC(3*nv, &q->rr);
    EMALLOC(  nt, &q->tt);

    q->nv = nv;
    q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_volume_fin(MeshVolume *q) {
    EFREE(q->rr);
    EFREE(q->tt);
    EFREE(q);
}

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
static double M0(double a[3], double b[3], double c[3]) {
    enum {X, Y, Z};
    return  (+a[X]*(b[Y]*c[Z]-b[Z]*c[Y])
             -a[Y]*(b[X]*c[Z]-b[Z]*c[X])
             +a[Z]*(b[X]*c[Y]-b[Y]*c[X]));
}
static void get(double *rr, int i, /**/ double *r1) {
    enum {X, Y, Z};
    double *r0;
    r0 = &rr[3*i];
    r1[X] = r0[X]; r1[Y] = r0[Y]; r1[Z] = r0[Z];
}
static double volume(int nt, int4 *tt, double *rr) {
    enum {X, Y, Z};
    int i, ia, ib, ic;
    double a[3], b[3], c[3], sum;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(get(rr, ia, /**/ a));
        UC(get(rr, ib, /**/ b));
        UC(get(rr, ic, /**/ c));
        kahan_sum_add(kahan_sum, M0(a, b, c));
    }
    sum = kahan_sum_get(kahan_sum);
    return sum/6;
}
double mesh_volume_apply0(MeshVolume *q, Vectors *positions) {
    int nv, nt, offset;
    double *rr;
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;
    UC(to_com(nv, offset, positions, /**/ rr));
    return volume(nt, tt, rr);
}

 void mesh_volume_apply(MeshVolume *q, int m, Vectors *positions, double *volume0) {
    int i, nt, nv, offset;
    double *rr;
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;
    for (i = 0; i < m; i++) {
        UC(to_com(nv, offset, positions, /**/ rr));
        UC(volume0[i] = volume(nt, tt, rr));
        offset += nv;
    }
}
/* end src/mesh/volume/imp/main.h */

/* end src/mesh/volume/imp.cpp */

#line 1 "src/mesh_bounce/imp.cu"
