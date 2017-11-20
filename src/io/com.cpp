#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector_types.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "mpi/glb.h"
#include "utils/mc.h"

#include "utils/halloc.h"
#include "utils/os.h"
#include "utils/error.h"

#include "msg.h"

enum {MAX_CHAR_PER_LINE = 128};

static void shift(float3 *r) {
    r->x = m::x2g(r->x);
    r->y = m::y2g(r->y);
    r->z = m::z2g(r->z);
}

static int swrite(int n, const int *ii, const float3 *rr, /**/ char *s) {
    int i, id, c, start;
    float3 r;
    for (i = start = 0; i < n; ++i) {
        id = ii[i];
        r  = rr[i];
        shift(/**/ &r);
        c = sprintf(s + start, "%d %g %g %g\n", id, r.x, r.y, r.z);
        if (c >= MAX_CHAR_PER_LINE)
            signal_error_extra("buffer too small : %d / %d", c, MAX_CHAR_PER_LINE);
        start += c;
    }
    return start;
}

static void write_mpi(const char *fname, long n, const char *data) {
    MPI_File f;
    MPI_Status status;
    MPI_Offset base, offset = 0;
    MPI_Offset len = n * sizeof(char);

    MC(MPI_File_open(m::cart, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f));
    MC(MPI_File_set_size(f, 0));
    MC(MPI_File_get_position(f, &base)); 

    MC( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, m::cart) );
    MC( MPI_File_write_at_all(f, base + offset, data, n, MPI_CHAR, &status) ); 
    MC( MPI_File_close(&f) );
}


void dump_com(long id, int n, const int *ii, const float3 *rr) {
    char fname[256] = {0}, *data;
    long nchar = 0;
    
    emalloc(MAX_CHAR_PER_LINE * n * sizeof(char), (void**) &data);

    if (m::rank == 0) os::mkdir(DUMP_BASE "/com");

    sprintf(fname, DUMP_BASE "/com/%04ld.txt", id);
    
    UC(nchar = swrite(n, ii, rr, /**/ data));
    write_mpi(fname, nchar, data);
    
    free(data);
}
