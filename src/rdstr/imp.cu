#include <stdio.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/q.h"
#include "d/api.h"
#include "d/ker.h"

#include "inc/def.h"
#include "msg.h"
#include "m.h"
#include "cc.h"
#include "l/m.h"

#include "inc/type.h"
#include "inc/mpi.h"
#include "inc/dev.h"
#include "mc.h"

#include "kl.h"

#include "minmax.h"

#include "mdstr/buf.h"
#include "mdstr/gen.h"

#include "rdstr/imp.h"
#include "rdstr/dev.h"

namespace rdstr {
namespace sub {

enum {X, Y, Z};

void waitall(MPI_Request rr[26]) {
    MPI_Status ss[26];
    l::m::Waitall(26, rr, ss) ;
}

void cancelall(MPI_Request rr[26]) {
    for (int i = 0; i < 26; ++i) MC(MPI_Cancel(rr + i));
}

void extents(const Particle *pp, int nc, int nv, /**/ float3 *ll, float3 *hh) {
    if (nc) minmax(pp, nv, nc, /**/ ll, hh);
}

void get_pos(int n, const float3 *ll, const float3 *hh, /**/ float *rr) {
    for (int i = 0; i < n; ++i) {
        float3 l = ll[i], h = hh[i];
        float *r = rr + 3 * i;
        r[X] = 0.5f * (l.x + h.x);
        r[Y] = 0.5f * (l.y + h.y);
        r[Z] = 0.5f * (l.z + h.z);
    }
}

void pack(int *reord[27], const int counts[27], const Particle *pp, int nv, /**/ Partbuf *bpp) {
    gen::pack <Particle, gen::Device> (reord, counts, pp, nv, /**/ bpp);
}

void post_send(int nv, const int counts[27], const Partbuf *bpp, MPI_Comm cart, int bt, int rnk_ne[27],
               /**/ MPI_Request sreq[26]) {
    dSync(); // wait for pack
    gen::post_send(nv, counts, bpp, cart, bt, rnk_ne, /**/ sreq);
}

void post_recv(MPI_Comm cart, int nmax, int bt, int ank_ne[27], /**/ Partbuf *bpp, MPI_Request rreq[26]) {
    gen::post_recv(cart, nmax, bt, ank_ne, /**/ bpp, rreq);
}

int unpack(int npd, const Partbuf *bpp, const int counts[27], /**/ Particle *pp) {
    return gen::unpack <Particle, gen::Device> (npd, bpp, counts, /**/ pp);
}

void shift(int npd, const int counts[27], /**/ Particle *pp) {
    int nm = counts[0]; /* skip bulk */
    for (int i = 1; i < 27; ++i) {
        int c = counts[i];
        int n = c * npd;
        KL(dev::shift, (k_cnf(n)), (n, i, /**/ pp + nm * npd));
        nm += c;
    }
}

} // sub
} // rdstr
