#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>

struct Coords;
void local2global(const Coords *, float3, float3 *);
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

enum {MAX_CHAR_PER_LINE = 128};
#define BASE DUMP_BASE "/diag/com"

static int swrite(const Coords *coords, int n, const int *ii, const float3 *rr, char *s) {
    int start = 0;
    for (int i = 0; i < n; ++i) {
        float3 v, rg;
        local2global(coords, rr[i], &rg);
        int c = sprintf(s + start, "%d %g %g %g %g %g %g\n",
                        ii[i], rg.x, rg.y, rg.z, v.x, v.y, v.z);
        if (c >= MAX_CHAR_PER_LINE) ERR("buffer too small : %d / %d", c, MAX_CHAR_PER_LINE);
        start += c;
    }
    return start;
}

static void write_mpi(MPI_Comm comm, const char *fname, long n, const char *data) {
    WriteFile *f;
    UC(write_file_open(comm, fname, &f));
    UC(write_all(comm, data, n, f));
    UC(write_file_close(f));
}

void io_com_dump(MPI_Comm comm, const Coords *coords, const char *name, long id,
                 int n, const int *ii, const float3 *rr) {
    char fname[FILENAME_MAX] = {0};
    char *data;
    EMALLOC(MAX_CHAR_PER_LINE * n, &data);
    if (m::is_master(comm)) UC(os_mkdir(BASE));
    MC(m::Barrier(comm));
    sprintf(fname, BASE "/%s.%04ld.txt", name, id);
    long nchar;
    UC(nchar = swrite(coords, n, ii, rr, data));
    write_mpi(comm, fname, nchar, data);
    EFREE(data);
}

#undef BASE
