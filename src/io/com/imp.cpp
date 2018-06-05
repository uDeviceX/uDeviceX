#include <stdio.h>
#include <vector_types.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "utils/mc.h"

#include "coords/imp.h"

#include "utils/imp.h"
#include "utils/os.h"
#include "utils/error.h"

#include "io/write/imp.h"

#include "imp.h"

enum {MAX_CHAR_PER_LINE = 128};

#define BASE DUMP_BASE "/diag/com"

static int swrite(const Coords *coords, int n, const int *ii, const float3 *rr, /**/ char *s) {
    int i, id, c, start;
    float3 r, v, rg;
    for (i = start = 0; i < n; ++i) {
        id = ii[i];
        r  = rr[i];
        local2global(coords, r, /**/ &rg);
        c = sprintf(s + start, "%d %g %g %g %g %g %g\n", id, rg.x, rg.y, rg.z, v.x, v.y, v.z);
        if (c >= MAX_CHAR_PER_LINE)
            ERR("buffer too small : %d / %d", c, MAX_CHAR_PER_LINE);
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

void io_com_dump(MPI_Comm comm, const Coords *coords, const char *name, long id, int n, const int *ii, const float3 *rr) {
    char fname[FILENAME_MAX] = {0}, *data;
    long nchar = 0;
    
    EMALLOC(MAX_CHAR_PER_LINE * n, &data);

    if (m::is_master(comm))
        UC(os_mkdir(BASE));

    sprintf(fname, BASE "/%s.%04ld.txt", name, id);
    
    UC(nchar = swrite(coords, n, ii, rr, /**/ data));
    write_mpi(comm, fname, nchar, data);
    
    EFREE(data);
}

#undef BASE
