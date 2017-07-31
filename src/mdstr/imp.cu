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
#include "mdstr/dev.h"

namespace mdstr {
namespace sub {

enum {X, Y, Z};

void waitall(MPI_Request rr[26]) {
    MPI_Status ss[26];
    l::m::Waitall(26, rr, ss) ;
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

// void pack(int *reord[27], const int counts[27], const Particle *pp, int nv, /**/ Particle *pps[27]) {
//     for (int fid = 0; fid < 27; ++fid)
//         for (int j = 0; j < counts[fid]; ++j) {
//             int src = reord[fid][j];
//             CC(cudaMemcpyAsync(pps[fid] + j * nv, pp + src * nv, nv * sizeof(Particle), D2H));
//         }
// }

// void post_send(int nv, const int counts[27], Particle *const pp[27], MPI_Comm cart, int btc, int btp, int rnk_ne[27],
//                /**/ MPI_Request sreqc[26], MPI_Request sreqp[26]) {
//     for (int i = 1; i < 27; ++i)
//         MC(l::m::Isend(counts + i, 1, MPI_INT, rnk_ne[i], btc + i, cart, sreqc + i - 1));

//     for (int i = 1; i < 27; ++i)
//         MC(l::m::Isend(pp[i], nv * counts[i], datatype::particle, rnk_ne[i], btp + i, cart, sreqp + i - 1));
// }

// void post_recv(MPI_Comm cart, int btc, int btp, int ank_ne[27],
//                /**/ int counts[27], Particle *pp[27], MPI_Request rreqc[26], MPI_Request rreqp[26]) {
//     for (int i = 1; i < 27; ++i)
//         MC(l::m::Irecv(counts + i, 1, MPI_INT, ank_ne[i], btc + i, cart, rreqc + i - 1));

//     for (int i = 1; i < 27; ++i)
//         MC(l::m::Irecv(pp[i], MAX_PART_NUM, datatype::particle, ank_ne[i], btp + i, cart, rreqp + i - 1));
// }

// int unpack(int nv, Particle *const ppr[27], const int counts[27], /**/ Particle *pp) {
//     int nm = 0;
//     for (int i = 0; i < 27; ++i) {
//         int c = counts[i];
//         int n = c * nv;
//         if (n) {
//             CC(cudaMemcpyAsync(pp + nm * nv, ppr[i], n * sizeof(Particle), H2D));
//             if (i) dev::shift <<<k_cnf(n)>>> (n, i, /**/ pp + nm * nv);
//         } 
//         nm += c;
//     }
//     return nm;
// }

} // sub
} // mdstr
