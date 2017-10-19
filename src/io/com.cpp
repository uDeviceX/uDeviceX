#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector_types.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "mpi/wrapper.h"
#include "mpi/glb.h"
// #include "utils/mc.h"

#include "utils/os.h"
#include "utils/error.h"

#include "msg.h"

enum {MAX_CHAR_PER_LINE = 128};

static int swrite(int n, const int *ii, const float3 *rr, /**/ char *s) {
    int i, id, c, start;
    float3 r;
    for (i = start = 0; i < n; ++i) {
        id = ii[i];
        r  = rr[i];
        c = sprintf(s + start, "%d %g %g %g\n", id, r.x, r.y, r.z);
        if (c >= MAX_CHAR_PER_LINE)
            signal_error_extra("buffer too small : %d / %d", c, MAX_CHAR_PER_LINE);
        start += c;
    }
    return start;
}

static void write(int n, const int *ii, const float3 *rr, /**/ FILE *f) {
    int i, id;
    float3 r;
    for (i = 0; i < n; ++i) {
        id = ii[i];
        r  = rr[i];
        fprintf(f, "%d %g %g %g\n", id, r.x, r.y, r.z);
    }
}

void dump_com(long id, int n, const int *ii, const float3 *rr) {
    char fname[256] = {0};
    FILE *f;
    
    if (m::rank == 0) {
        os::mkdir(DUMP_BASE "/com");
        sprintf(fname, DUMP_BASE "/com/%04ld.txt", id);
        f = fopen(fname, "w");
        write(n, ii, rr, /**/ f);
        fclose(f);
    } else {
        ERR("Not implemented yet\n");
    }
}
