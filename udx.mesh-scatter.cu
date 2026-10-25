// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/mesh/scatter/imp.cpp */
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
/* begin src/algo/scalars/imp.h */
struct Particle;
struct Scalars;
struct Vectors;

// tag::interface[]
void scalars_float_ini(int n, const float*, /**/ Scalars**);
void scalars_double_ini(int n, const double*, /**/ Scalars**);
void scalars_vectors_ini(int n, const Vectors*, int dim, /**/ Scalars**);

void scalars_zero_ini(int n, /**/ Scalars**);
void scalars_one_ini(int n, /**/ Scalars**);

void scalars_fin(Scalars*);
double scalars_get(const Scalars*, int i);
// end::interface[]
/* end src/algo/scalars/imp.h */

/* begin src/mesh/scatter/imp.h */
struct MeshScatter;
struct MeshRead;
struct Scalars;

// tag::interface[]
void mesh_scatter_ini(MeshRead*, MeshScatter**); // <1>
void mesh_scatter_fin(MeshScatter*);
void mesh_scatter_edg2vert(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
void mesh_scatter_edg2vert_avg(MeshScatter*, int nm, Scalars*, /**/ double*); // <2>
// end::interface[]
/* end src/mesh/scatter/imp.h */
/* begin src/mesh/scatter/imp/type.h */
struct MeshScatter {
    int ne, nv;
    int *ee;
    int *deg; /* vert. degree */
};
/* end src/mesh/scatter/imp/type.h */
/* begin src/mesh/scatter/imp/main.h */
static void deg_ini(int nv, int ne, const int *ee, /**/ int *deg) {
    int i, k, a, b;
    for (i = 0; i < nv; i++) deg[i] = 0;
    for (i = k = 0; i < ne; i++) {
        a = ee[k++];
        b = ee[k++];
        deg[a]++;
        deg[b]++;
    }
}
void mesh_scatter_ini(MeshRead *mesh, MeshScatter **pq) {
    MeshScatter *q;
    int i, j, nv, ne;
    int *ee, *deg;
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
    EMALLOC(  nv, &deg);
    UC(deg_ini(nv, ne, ee, /**/ deg));

    q->nv = nv; q->ne = ne;
    q->ee = ee; q->deg = deg;
    *pq = q;
}

static void edg2vert(MeshScatter *q, Scalars *sc, int offset_v, int offset_e, /**/ double *o) {
    int ne, i, k, a, b;
    double value;
    const int *ee;
    ne = q->ne;
    ee = q->ee;
    for (k = i = 0; i < ne; i++) {
        a = ee[k++]; b = ee[k++];
        UC(value = scalars_get(sc, i + offset_e));
        o[a + offset_v] += value;
        o[b + offset_v] += value;
    }
}
void mesh_scatter_edg2vert_avg(MeshScatter *q, int nm, Scalars *sc, /**/ double *o) {
    int i, j, k, d, n, nv, ne;
    int offset_v, offset_e;
    nv = q->nv;
    ne = q->ne;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(edg2vert(q, sc, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
    for (i = j = 0; i < nm; i++)
        for (k = 0; k < n; k++) {
            d = q->deg[k];
            if (d <= 0) ERR("wrong vert. degree: %d", d);
            if (j >= n) ERR("j=%d >= n=%d", j, n);
            o[j++] /= d;
        }
}

void mesh_scatter_edg2vert(MeshScatter *q, int nm, Scalars *sc, /**/ double *o) {
    int i, n, nv, ne;
    int offset_v, offset_e;
    nv = q->nv;
    ne = q->ne;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(edg2vert(q, sc, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
}

void mesh_scatter_fin(MeshScatter *q) {
    EFREE(q->ee);
    EFREE(q->deg);
    EFREE(q);
}
/* end src/mesh/scatter/imp/main.h */
/* end src/mesh/scatter/imp.cpp */

#line 1 "src/mesh/spherical/imp.cpp"
