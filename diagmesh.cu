#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>

struct Particle;
struct MeshRead;
void mesh_read_fin(MeshRead *);
int mesh_read_get_nt(const MeshRead *);
int mesh_read_get_nv(const MeshRead *);
const int4 *mesh_read_get_tri(const MeshRead *);
void msg_print(const char *, ...);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);
void *ememcpy(void *, const void *, size_t);
void efopen(const char *, const char *, FILE **);
void efclose(FILE *);

struct DiagMesh {
    char path[FILENAME_MAX];
    int nv, nt;
    int4 *tt;
};

void diag_mesh_ini(const char *path, MeshRead *mesh, DiagMesh **pq) {
    FILE *f;
    DiagMesh *q;
    EMALLOC(1, &q);
    *pq = q;
    q->nv = mesh_read_get_nv(mesh);
    q->nt = mesh_read_get_nt(mesh);
    EMALLOC(q->nt, &q->tt);
    EMEMCPY(q->nt, mesh_read_get_tri(mesh), q->tt);
    snprintf(q->path, FILENAME_MAX, "%s/%s", DUMP_BASE, path);
    UC(efopen(q->path, "w", &f));
    UC(efclose(f));
    msg_print("DiagMesh: %s", q->path);
}

void diag_mesh_fin(DiagMesh *q) {
    EFREE(q->tt);
    EFREE(q);
}

void diag_mesh_apply(DiagMesh *, MPI_Comm, float, int, Particle *) {}
