// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/mesh/edg_len/imp.cpp */
#include <stdio.h>
#include <math.h>
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

/* begin src/mesh/edg_len/imp.h */
struct MeshEdgLen;
struct MeshRead;
struct Vectors;

// tag::interface[]
void mesh_edg_len_ini(MeshRead*, MeshEdgLen**); // <1>
void mesh_edg_len_fin(MeshEdgLen*);
void mesh_edg_len_apply(MeshEdgLen*, int nm, Vectors*, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/edg_len/imp.h */
/* begin src/mesh/edg_len/imp/type.h */
struct MeshEdgLen {
    int ne, nv;
    int *ee;
};
/* end src/mesh/edg_len/imp/type.h */
/* begin src/mesh/edg_len/imp/main.h */
void mesh_edg_len_ini(MeshRead *mesh, MeshEdgLen **pq) {
    MeshEdgLen *q;
    int i, j, nv, ne, *ee;
    const int4 *dd;
    int4 d;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    dd = mesh_read_get_dih(mesh);
    EMALLOC(2*ne, &ee);
    for (j = i = 0; i < ne; i++) {
        d = dd[i];
        ee[j++] = d.y; ee[j++] = d.z;
    }
    q->nv = nv; q->ne = ne; q->ee = ee;
    *pq = q;
}

static void mesh_edg_len_get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double sum_sq(double x, double y, double z) { return x*x + y*y + z*z; }
static double len0(Vectors *pos, int i, int j) {
    enum {X, Y, Z};
    double d, a[3], b[3];
    mesh_edg_len_get(pos, i, /**/ a);
    mesh_edg_len_get(pos, j, /**/ b);
    d = sum_sq(a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]);
    return sqrt(d);
}
static void len(MeshEdgLen *q, Vectors *pos, int offset_v, int offset_e, /**/ double *o) {
    int ne, i, k, a, b;
    double value;
    const int *ee;
    ne = q->ne;
    ee = q->ee;
    for (k = i = 0; i < ne; i++) {
        a = ee[k++]; b = ee[k++];
        UC(value = len0(pos, a + offset_v, b + offset_v));
        o[i + offset_e] = value;
    }
}
void mesh_edg_len_apply(MeshEdgLen *q, int nm, Vectors *pos, /**/ double *o) {
    int i, nv, ne;
    int offset_v, offset_e; /* vert and edge */
    nv = q->nv;
    ne = q->ne;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(len(q, pos, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
}

void mesh_edg_len_fin(MeshEdgLen *q) {
    EFREE(q->ee);
    EFREE(q);
}
/* end src/mesh/edg_len/imp/main.h */
/* end src/mesh/edg_len/imp.cpp */

#line 1 "src/mesh/eng_julicher/imp.cpp"
