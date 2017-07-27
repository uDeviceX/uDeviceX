#include <mpi.h>
#include "m.h"
#include "l/m.h"

#include <cstdio>
#include "common.h"
#include "common.mpi.h"
#include "common.cuda.h"
#include <conf.h>

#include "mdstr/imp.h"
#include "mdstr/ini.h"

namespace mdstr {
namespace sub {

enum {X, Y, Z};

static int r2fid(const float r[3]) {
    int cx, cy, cz;
    cx = (2 + (r[X] >= -XS / 2) + (r[X] >= XS / 2)) % 3;
    cy = (2 + (r[Y] >= -YS / 2) + (r[Y] >= YS / 2)) % 3;
    cz = (2 + (r[Z] >= -ZS / 2) + (r[Z] >= ZS / 2)) % 3;
    return cx + 3 * cy + 9 * cz;
}

void get_dests(const float *rr, int nm, /**/ int *dests[27], int counts[27]) {
    int i, fid, did;
    for (i = 0; i < 27; ++i) counts[i] = 0;
    for (i = 0; i < nm; ++i) {
        fid = r2fid(rr + 3 * i);
        did = counts[fid] ++;
        dests[fid][did] = i;
    }
}

void pack(const int *dests[27], const int counts[27], const Particle *pp, int nm, int nv, /**/ Particle *pps[27]) {
    for (int fid = 0; fid < 27; ++fid)
        for (int j = 0; j < counts[fid]; ++j) {
            int src = dests[fid][j];
            CC(cudaMemcpyAsync(pps[fid] + j * nv, pp + src * nv, nv * sizeof(Particle), D2H));
        }
}

void post_send(const int counts[27], const Particle *pp[27], MPI_Comm cart, int btc, int btp, int rnk_ne[27],
               /**/ MPI_Request sreqc[26], MPI_Request sreqp[26]) {
    for (int i = 1; i < 27; ++i)
        MC(l::m::Isend(counts + i, 1, MPI_INT, rnk_ne[i], btc + i, cart, sreqc + i - 1));

    for (int i = 1; i < 27; ++i)
        MC(l::m::Isend(pp[i], counts[i], datatype::particle, rnk_ne[i], btp + i, cart, sreqp + i - 1));
}

void wait() {

}

void unpack() {

}

} // sub
} // mdstr
