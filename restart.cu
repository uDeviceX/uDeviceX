#include "memory.cuh"
#define DEV_CUDA
#define CC_RELEASE

#include "config.cuh"
#include "macros.cuh"
#include "types.cuh"
#include "constants.cuh"

#include <mpi.h>
#include <stdio.h>
#include <string.h>
void msg_print(const char *, ...);

void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);

namespace m {
bool is_master(MPI_Comm);
}

struct Coords;
struct Particle;
struct Rigid;

enum { RESTART_TEMPL = -1, RESTART_FINAL = -2 };
enum { RESTART_BEGIN = RESTART_FINAL };

typedef void (*StreamWriter)(const void *, FILE *);
typedef void (*StreamReader)(FILE *, void *);

BopStatus bop_ini(BopData **);
BopStatus bop_fin(BopData *);
BopStatus bop_set_n(long, BopData *);
BopStatus bop_set_type(BopType, BopData *);
BopStatus bop_set_vars(int, const char *, BopData *);
BopStatus bop_alloc(BopData *);
BopStatus bop_write_header(MPI_Comm, const char *, const BopData *);
BopStatus bop_write_values(MPI_Comm, const char *, const BopData *);
BopStatus bop_read_header(MPI_Comm, const char *, BopData *, char *);
BopStatus bop_read_values(MPI_Comm, const char *, BopData *);
BopStatus bop_get_n(const BopData *, long *);
void *bop_get_data(BopData *);

#define PATTERN "%s/%s/%s"


enum { BS = 1024 };


#define CSPR(a) do {                                        \
        int check = a;                                      \
        if (check < 0 || check >= BS-1)                     \
        ERR("Buffer too small to handle this format\n");    \
    } while (0)

static void id2code(const int id, char *code) {
    switch (id) {
    case RESTART_TEMPL:
        CSPR(sprintf(code, "templ"));
        break;
    case RESTART_FINAL:
        CSPR(sprintf(code, "final"));
        break;
    default:
        CSPR(sprintf(code, "%05d", id));
        break;
    }
}

static void gen_base_name(const char *pattern, const char *base, const char *code, const int id,      char *name) {
    char idcode[BS] = {0};
    id2code(id,      idcode);
    CSPR(sprintf(name, pattern, base, code, idcode));
}

static void gen_base_name_dump(const char *base, const char *code, const int id,      char *name) {
    gen_base_name(PATTERN, base, code, id, name);
}

static void gen_base_name_read(const char *base, const char *code, const int id,      char *name) {
    gen_base_name(PATTERN ".bop", base, code, id, name);
}

template <typename T>
static void write(MPI_Comm comm, const char *base, const char *code, int id, long n, const T *tt, BopType type, int nvars, const char *vars) {
    char name[BS];
    BopData *bop;
    T *tt_bop;
    gen_base_name_dump(base, code, id,      name);

    BPC(bop_ini(&bop));
    BPC(bop_set_n(n, bop));
    BPC(bop_set_type(type, bop));
    BPC(bop_set_vars(nvars, vars, bop));
    BPC(bop_alloc(bop));

    tt_bop = (T*) bop_get_data(bop);
    memcpy(tt_bop, tt, n * sizeof(T));

    BPC(bop_write_header(comm, name, bop));
    BPC(bop_write_values(comm, name, bop));

    BPC(bop_fin(bop));
}

template <typename T>
static void read(MPI_Comm comm, const char *base, const char *code, const int id, int *n, T *tt) {
    long np = 0;
    char hname[BS], dname[BS];
    BopData *bop;
    const T *tt_bop;
    gen_base_name_read(base, code, id,      hname);
    msg_print("reading '%s'", hname);

    BPC(bop_ini(&bop));
    BPC(bop_read_header(comm, hname, bop, dname));
    BPC(bop_alloc(bop));
    BPC(bop_read_values(comm, dname, bop));

    BPC(bop_get_n(bop, &np));
    tt_bop = (const T*) bop_get_data(bop);
    memcpy(tt, tt_bop, np * sizeof(T));
    *n = np;

    BPC(bop_fin(bop));
}

void restart_write_pp(MPI_Comm comm, const char *base, const char *code, int id, long n, const Particle *pp) {
    write(comm, base, code, id, n, pp, BopFLOAT, 6, "x y z vx vy vz");
}

void restart_read_pp(MPI_Comm comm, const char *base, const char *code, int id, int *n, Particle *pp) {
    read(comm, base, code, id, n, pp);
}

void restart_write_ii(MPI_Comm comm, const char *base, const char *code, int id, long n, const int *ii) {
    write(comm, base, code, id, n, ii, BopINT, 1, "i");
}

void restart_read_ii(MPI_Comm comm, const char *base, const char *code, int id, int *n, int *ii) {
    read(comm, base, code, id, n, ii);
}

void restart_write_ss(MPI_Comm comm, const char *base, const char *code, int id, long n, const Rigid *ss) {
    static const char *vars =
        "Ixx Ixy Ixz Iyy Iyz Izz "
        "mass x y z vx vy vz omx omy omz "
        "e0x e0y e0z e1x e1y e1z e2x e2y e2z "
        "fx fy fz tx ty tz i";
    enum {NVARS=32};
    write(comm, base, code, id, n, ss, BopFLOAT, NVARS, vars);
}

void restart_read_ss(MPI_Comm comm, const char *base, const char *code, int id, int *n, Rigid *ss) {
    read(comm, base, code, id, n, ss);
}

void restart_write_stream_one_node(MPI_Comm comm, const char *base, const char *code, int id, const void *data, StreamWriter sw) {
    char name[BS];
    FILE *f;
    if (!m::is_master(comm)) return;
    gen_base_name(PATTERN ".dat", base, code, id, name);
    UC(efopen(name, "w", &f));
    UC(sw(data, f));
    UC(efclose(f));
}

void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader sr, void *data) {
    char name[BS];
    FILE *f;
    gen_base_name(PATTERN ".dat", base, code, id, name);
    msg_print("reading '%s'", name);
    UC(efopen(name, "r", &f));
    UC(sr(f, data));
    UC(efclose(f));
}

#undef PATTERN

