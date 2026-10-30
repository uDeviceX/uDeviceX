#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

void msg_print(const char *, ...);
void os_mkdir(const char *);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
namespace m { bool is_master(MPI_Comm); }
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#ifndef UDX_SIM_OBJECTS_TYPE_COMMON
#define UDX_SIM_OBJECTS_TYPE_COMMON
#include "types.cuh"
#endif
#endif

struct int4;
struct MeshRead;

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
struct Particle;
struct Vectors;
struct Coords;

void vectors_float_ini(int n, const float*,      Vectors**);
void vectors_postions_ini(int n, const Particle*,      Vectors**);
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*,      Vectors**);
void vectors_postions_center_ini(const Coords*, int n, const Particle*,      Vectors**);
void vectors_velocities_ini(int n, const Particle*,      Vectors**);
void vectors_zero_ini(int n,      Vectors**);

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i,      float r[3]);

struct WriteFile;

int write_file_open(MPI_Comm, const char *path,      WriteFile**);
int write_file_close(WriteFile*);

void write_all(MPI_Comm, const void *const, int nbytes, WriteFile*);
int write_master(MPI_Comm, const void *const, int nbytes, WriteFile*);

int write_shift_indices(MPI_Comm, int,      int*);
int write_reduce(MPI_Comm, int,      int*);

struct Coords;
struct Particle;
struct int4;
struct MeshWrite;
struct MeshRead;
struct Vectors;

#ifndef UDX_SIM_MESH_WRITE_COMMON
#define UDX_SIM_MESH_WRITE_COMMON
enum {MESH_WRITE_EDGE, MESH_WRITE_CENTER};
#endif

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory,      MeshWrite**);
void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead*, const char *directory,      MeshWrite**);

void mesh_write_particles(MeshWrite*, MPI_Comm, const Coords*, int nm, const Particle*, int id);
void mesh_write_vectors(MeshWrite*, MPI_Comm, int nm, Vectors *pos, Vectors *vel, int id);

void mesh_write_fin(MeshWrite*);
struct MeshWrite {
    int4 *tt;
    int nv, nt;
    char directory[FILENAME_MAX];
    int shift_type;
};

#ifndef UDX_IO_OUTPUT_COMMON_UTIL
#define UDX_IO_OUTPUT_COMMON_UTIL
static char *cpy(char *dest, const char *src) { return strncpy(dest, src, FILENAME_MAX); }
static char *cat(char *dest, const char *src) { return strncat(dest, src, FILENAME_MAX); }
#endif

static void mkdir0(const char *p, const char *s) {
    char path[FILENAME_MAX] = {'\0'};
    cpy(path, p);
    cat(path, "/");
    cat(path, s);
    msg_print("mkdir -p '%s'", path);
    UC(os_mkdir(path));
}

#ifndef UDX_IO_OUTPUT_MKDIR_COMMON
#define UDX_IO_OUTPUT_MKDIR_COMMON
static void mkdir(MPI_Comm comm, const char *p, const char *s) {
    if (m::is_master(comm)) UC(mkdir0(p, s));
}
#endif
#define PATTERN "%s/%s/%05d.ply"
enum { NVP = 3                                   };

static void ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory,      MeshWrite **pq) {
    int i;
    MeshWrite *q;
    EMALLOC(1, &q);

    q->nv = nv; q->nt = nt;
    cpy(q->directory, directory);
    UC(mkdir(comm, DUMP_BASE, directory));
    EMALLOC(nt, &q->tt);
    for (i = 0; i < nt; i++)
        q->tt[i] = tt[i];
    q->shift_type = shifttype;
    *pq = q;
}

void mesh_write_ini(MPI_Comm comm, int shifttype, const int4 *tt, int nv, int nt, const char *directory,      MeshWrite **pq) {
    UC(ini(comm, shifttype, tt, nv, nt, directory,      pq));
}

void mesh_write_ini_from_mesh(MPI_Comm comm, int shifttype, MeshRead *cell, const char *directory,      MeshWrite **pq) {
    int nv, nt;
    const int4 *tt;
    nv = mesh_read_get_nv(cell);
    nt = mesh_read_get_nt(cell);
    tt = mesh_read_get_tri(cell);
    UC(ini(comm, shifttype, tt, nv, nt, directory,      pq));
}

void mesh_write_fin(MeshWrite *q) {
    EFREE(q->tt);
    EFREE(q);
}

static void header(MPI_Comm comm, int nc0, int nv, int nt, WriteFile *f) {
    int nc;
    int sz = 0;
    char s[BUFSIZ];
    UC(write_reduce(comm, nc0, &nc));
    if (m::is_master(comm))
        sz = snprintf(s, BUFSIZ - 1,
                     "ply\n"
                     "format binary_little_endian 1.0\n"
                     "element vertex %d\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property float u\nproperty float v\nproperty float w\n"
                     "element face  %d\n"
                     "property list int int vertex_index\n"
                     "end_header\n", nv*nc, nt*nc);
    write_master(comm, s, sz, f);
}

static void vert(MPI_Comm cart, int n, const Vectors *pos, const Vectors *vel, WriteFile *f) {
    float *D, *r, *v;
    int i, j, m;
    m = (3 + 3) * n;
    EMALLOC(m, &D);
    for (i = j = 0; i < n; i++) {
        r = &D[j]; j += 3;
        v = &D[j]; j += 3;
        UC(vectors_get(pos, i,      r));
        UC(vectors_get(vel, i,      v));
    }
    UC(write_all(cart, D, m*sizeof(D[0]), f));
    EFREE(D);
}

static void wfaces0(MPI_Comm comm, int *buf, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {

    int c, t, b;
    int n, shift;
    int4 tri;
    n = nc * nv;
    UC(write_shift_indices(comm, n, &shift));
    for (b = c = 0; c < nc; ++c)
        for (t = 0; t < nt; ++t) {
            tri = faces[t];
            buf[b++] = NVP;
            buf[b++] = shift + nv*c + tri.x;
            buf[b++] = shift + nv*c + tri.y;
            buf[b++] = shift + nv*c + tri.z;
        }
    UC(write_all(comm, buf, b*sizeof(buf[0]), f));
}

static void wfaces(MPI_Comm cart, const int4 *faces, int nc, int nv, int nt, WriteFile *f) {
    int *buf;
    int sz;
    sz = (1 + NVP) * nc * nt;
    EMALLOC(sz, &buf);
    UC(wfaces0(cart, buf, faces, nc, nv, nt, f));
    EFREE(buf);
}

static void mesh_write(MPI_Comm comm, int nc, int nv, int nt,
                        const Vectors *pos, const Vectors *vel, const int4 *faces,
                        WriteFile *f) {
    int n;
    n = nc * nv;
    UC(header(comm, nc, nv, nt, f));
    UC(vert(comm, n, pos, vel, f));
    UC(wfaces(comm, faces, nc, nv, nt, f));
}

void mesh_write_particles(MeshWrite *q, MPI_Comm comm, const Coords *coords, int nc, const Particle *pp, int id) {
    int nv, nt, n;
    const int4 *tt;
    Vectors *pos, *vel;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX] = {'\0'} ;

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    n = nv * nc;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("sprintf failed");
    UC(write_file_open(comm, path,      &f));
    switch (q->shift_type) {
    case MESH_WRITE_EDGE:
        vectors_postions_edge_ini(coords, n, pp,      &pos);
        break;
    case MESH_WRITE_CENTER:
        vectors_postions_center_ini(coords, n, pp,      &pos);
        break;
    default:
        ERR("unkown q->type: %d", q->shift_type);
    }
    vectors_velocities_ini(n, pp,      &vel);
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    vectors_fin(pos);
    vectors_fin(vel);
    UC(write_file_close(f));
}

void mesh_write_vectors(MeshWrite *q, MPI_Comm comm, int nc, Vectors *pos, Vectors *vel, int id) {
    int nv, nt;
    const int4 *tt;
    WriteFile *f;
    const char *directory;
    char path[FILENAME_MAX] = {'\0'};

    nv = q->nv; nt = q->nt; tt = q->tt; directory = q->directory;
    if (snprintf(path, FILENAME_MAX, PATTERN, DUMP_BASE, directory, id) < 0)
        ERR("snprintf failed");
    UC(write_file_open(comm, path,      &f));
    UC(mesh_write(comm, nc, nv, nt, pos, vel, tt, f));
    UC(write_file_close(f));
}
