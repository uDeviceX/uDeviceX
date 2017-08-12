#include <mpi.h>
#include <cstdio>
#include <conf.h>
#include "inc/conf.h"

#include "common.h"
#include "m.h"
#include "cc.h"

#include "l/m.h"
#include "inc/type.h"
#include "common.mpi.h"
#include "inc/dev.h"

#include "mdstr/imp.h"
#include "mdstr/ini.h"
#include "mdstr/dev.h"

namespace mdstr {
namespace sub {

enum {X, Y, Z};

void waitall(MPI_Request rr[26]) {
    MPI_Status ss[26];
    MC(l::m::Waitall(26, rr, ss));
}

void cancelall(MPI_Request rr[26]) {
    for (int i = 0; i < 26; ++i) MC(MPI_Cancel(rr + i));
}


static int r2fid(const float r[3]) {
    int cx, cy, cz;
    cx = (2 + (r[X] >= -XS / 2) + (r[X] >= XS / 2)) % 3;
    cy = (2 + (r[Y] >= -YS / 2) + (r[Y] >= YS / 2)) % 3;
    cz = (2 + (r[Z] >= -ZS / 2) + (r[Z] >= ZS / 2)) % 3;
    return cx + 3 * cy + 9 * cz;
}

void get_reord(const float *rr, int nm, /**/ int *reord[27], int counts[27]) {
    int i, fid, did;
    for (i = 0; i < 27; ++i) counts[i] = 0;
    for (i = 0; i < nm; ++i) {
        fid = r2fid(rr + 3 * i);
        did = counts[fid] ++;
        reord[fid][did] = i;
    }
}

void post_sendc(const int counts[27], MPI_Comm cart, int btc, int rnk_ne[27], /**/ MPI_Request sreqc[26]) {
    for (int i = 1; i < 27; ++i)
        MC(l::m::Isend(counts + i, 1, MPI_INT, rnk_ne[i], btc + i, cart, sreqc + i - 1));
}

void post_recvc(MPI_Comm cart, int btc, int ank_ne[27], /**/ int rcounts[27], MPI_Request rreqc[26]) {
    for (int i = 1; i < 27; ++i)
        MC(l::m::Irecv(rcounts + i, 1, MPI_INT, ank_ne[i], btc + i, cart, rreqc + i - 1));
}

} // sub
} // mdstr
