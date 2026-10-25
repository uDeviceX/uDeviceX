// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/io/vtk/mesh/imp.cpp */
#include <stdio.h>
#include <vector_types.h>

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

/* begin src/io/vtk/mesh/imp.h */
struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*, /**/ Mesh**);
void mesh_copy(const Mesh*, /**/ Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
int mesh_nt(Mesh*);
int mesh_ne(Mesh*);
const int* mesh_tt(Mesh*);
/* end src/io/vtk/mesh/imp.h */

/* begin src/io/vtk/mesh/imp/type.h */
struct Mesh {
    int nv, nt, ne; /* vertices and triangles */
    int *tt;   /* triangles [v0 v1 v2] [v0 v1 v2] .. */
    int *ee;   /* edges [e0 e1] [e0 e1] .. */
};
/* end src/io/vtk/mesh/imp/type.h */
/* begin src/io/vtk/mesh/imp/main.h */
void mesh_ini(MeshRead *mesh, /**/ Mesh **pq) {
    Mesh *q;
    int i, j, nv, nt, ne;
    const int4 *tt, *dd;
    int4 t, d;
    EMALLOC(1, &q);

    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    ne = mesh_read_get_ne(mesh);
    tt = mesh_read_get_tri(mesh);
    dd = mesh_read_get_dih(mesh);

    EMALLOC(3*nt, &q->tt);
    EMALLOC(2*ne, &q->ee);
    for (i = j = 0; i < nt; i++) {
        t = tt[i];
        q->tt[j++] = t.x; q->tt[j++] = t.y; q->tt[j++] = t.z;
    }

    for (i = j = 0; i < ne; i++) {
        d = dd[i];
        q->ee[j++] = d.y; q->ee[j++] = d.z;
    }

    q->nv = nv; q->nt = nt; q->ne = ne;
    *pq = q;
}

void mesh_copy(const Mesh *f, /**/ Mesh **pt) {
    Mesh *t; /* to, from */
    int nv, nt, ne;
    EMALLOC(1, &t);

    nv = f->nv; nt = f->nt; ne = f->ne;

    EMALLOC(3*nt, &t->tt);
    EMALLOC(2*ne, &t->ee);

    EMEMCPY(3*nt, f->tt, t->tt);
    EMEMCPY(2*ne, f->ee, t->ee);

    t->nv = nv; t->nt = nt; t->ne = ne;
    *pt = t;
}

void mesh_fin(Mesh *q) {
    EFREE(q->tt);
    EFREE(q->ee);
    EFREE(q);
}

int mesh_nv(Mesh *q) { return q->nv; }
int mesh_nt(Mesh *q) { return q->nt; }
int mesh_ne(Mesh *q) { return q->ne; }
const int* mesh_tt(Mesh *q) { return q->tt; }
/* end src/io/vtk/mesh/imp/main.h */
/* end src/io/vtk/mesh/imp.cpp */

#line 1 "src/io/write/imp.cpp"
