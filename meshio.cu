#define DEV_CUDA
#define CC_RELEASE

#include "config.cuh"
#include "macros.cuh"
#include "constants.cuh"

#include <vector_types.h>

void emalloc(size_t, void **);
void efree(void *);
void *ememcpy(void *, const void *, size_t);

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

struct MeshRead;
void mesh_read_ini_off(const char *, MeshRead **);
void mesh_read_ini_ply(const char *, MeshRead **);
void mesh_read_fin(MeshRead *);
int mesh_read_get_nt(const MeshRead *);
int mesh_read_get_nv(const MeshRead *);
int mesh_read_get_ne(const MeshRead *);
const int4 *mesh_read_get_tri(const MeshRead *);
const int4 *mesh_read_get_dih(MeshRead *);

struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*,      Mesh**);
void mesh_copy(const Mesh*,      Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
int mesh_nt(Mesh*);
int mesh_ne(Mesh*);
const int* mesh_tt(Mesh*);

struct Mesh {
    int nv, nt, ne;
    int *tt;
    int *ee;
};
void mesh_ini(MeshRead *mesh,      Mesh **pq) {
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

void mesh_copy(const Mesh *f,      Mesh **pt) {
    Mesh *t;
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




