#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include "types.cuh"
#include <mpi.h>
#include <stdio.h>

struct Coords;
float xl2xg(const Coords *, float);
float yl2yg(const Coords *, float);
float zl2zg(const Coords *, float);
void os_mkdir(const char *);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);
struct WriteFile;
int write_file_open(MPI_Comm, const char *, WriteFile **);
int write_file_close(WriteFile *);
void write_all(MPI_Comm, const void *, int, WriteFile *);
namespace m {
bool is_master(MPI_Comm);
int Barrier(MPI_Comm);
}
namespace mpicheck { void check(int, const char *, int); }
#define MC(ans) do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

#define BASE DUMP_BASE "/diag/rig"

static int swrite_v(const float v[3], char *buf) {
    return sprintf(buf, "%+.6e %+.6e %+.6e ", v[0], v[1], v[2]);
}

static int swrite(const Coords *c, int ns, const Rigid *ss, int maxn, char *buf) {
    int nchar = 0;
    for (int i = 0; i < ns; ++i) {
        const Rigid *s = &ss[i];
        float com[3] = {xl2xg(c, s->com[0]), yl2yg(c, s->com[1]), zl2zg(c, s->com[2])};
        nchar += sprintf(buf + nchar, "%04d ", (int)s->id);
        nchar += swrite_v(com, buf + nchar);
        nchar += swrite_v(s->v, buf + nchar);
        nchar += swrite_v(s->om, buf + nchar);
        nchar += swrite_v(s->fo, buf + nchar);
        nchar += swrite_v(s->to, buf + nchar);
        nchar += swrite_v(s->e0, buf + nchar);
        nchar += swrite_v(s->e1, buf + nchar);
        nchar += swrite_v(s->e2, buf + nchar);
        nchar += sprintf(buf + nchar, "\n");
        if (nchar > maxn) ERR("exceed buffer capacity: [%d / %d]", nchar, maxn);
    }
    return nchar;
}

static void write_mpi(MPI_Comm comm, const char *fname, long n, const char *data) {
    WriteFile *f;
    UC(write_file_open(comm, fname, &f));
    UC(write_all(comm, data, n, f));
    UC(write_file_close(f));
}

void io_rig_dump(MPI_Comm comm, const Coords *c, const char *name, long id,
                 int ns, const Rigid *ss) {
    enum {NC_ID = 5, NC_END = 1, NC_v3 = 42, NC_SECURITY = 1, N_v3 = 8};
    int maxnc = ns * (NC_ID + NC_END + NC_v3 * N_v3 + NC_SECURITY);
    char fname[FILENAME_MAX];
    char *data;
    EMALLOC(maxnc, &data);
    if (m::is_master(comm)) UC(os_mkdir(BASE));
    MC(m::Barrier(comm));
    sprintf(fname, BASE "/%s.%04ld.txt", name, id);
    int nchar = swrite(c, ns, ss, maxnc, data);
    write_mpi(comm, fname, nchar, data);
    EFREE(data);
}

#undef BASE
