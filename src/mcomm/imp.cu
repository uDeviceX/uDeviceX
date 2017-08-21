#include <vector>
#include <cassert>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "d/q.h"
#include "d/ker.h"

#include "m.h"
#include "common.h"
#include "msg.h"
#include "cc.h"
#include "l/m.h"

#include "inc/type.h"
#include "inc/mpi.h"
#include "mc.h"
#include "inc/dev.h"

#include "kl.h"

#include "mcomm/type.h"
#include "mcomm/imp.h"
#include "mcomm/dev.h"
#include "mcomm/ini.h"
#include "mcomm/fin.h"

namespace mcomm {
namespace sub {

void cancel_req(Reqs *r) {
    for (int i = 0; i < 26; ++i) {
        MC(MPI_Cancel(r->pp     + i));
        MC(MPI_Cancel(r->counts + i));
    }
}

void wait_req(Reqs *r) {
    MPI_Status ss[26];
    MC(l::m::Waitall(26, r->pp,     ss));
    MC(l::m::Waitall(26, r->counts, ss));
}

enum {X, Y, Z};
enum {BULK, FACE, EDGE, CORN};

#define i2del(i) {((i) + 1) % 3 - 1,            \
            ((i) / 3 + 1) % 3 - 1,              \
            ((i) / 9 + 1) % 3 - 1}

static inline int dc2vc(const int d) {return (3 + d) % 3;}

static int d2i(const int d[3]) {
    return dc2vc(d[X]) + 3 * (dc2vc(d[Y]) + 3 * dc2vc(d[Z]));
}

namespace travel {
static void faces(const int i, const int d[3], /**/ std::vector<int> travellers[27]) {
    for (int c = 0; c < 3; ++c) {
        if (d[c]) {
            int df[3] = {0, 0, 0}; df[c] = d[c];
            int code = d2i(df);
            travellers[code].push_back(i);
        }
    }
}

static void edges(const int i, const int d[3], /**/ std::vector<int> travellers[27]) {
    for (int c = 0; c < 3; ++c) {
        int de[3] = {d[X], d[Y], d[Z]}; de[c] = 0;
        if (de[(c + 1) % 3] && de[(c + 2) % 3]) {
            int code = d2i(de);
            travellers[code].push_back(i);
        }
    }
}

static void cornr(const int i, const int d[3], /**/ std::vector<int> travellers[27]) {
    assert(d[X] && d[Y] && d[Z]);
    int code = d2i(d);
    travellers[code].push_back(i);
}
} // travel

int map(const float3* minext_hst, const float3 *maxext_hst, const int nm, /**/ std::vector<int> travellers[27], int counts[27]) {
    int i, d[3], type;
    
    for (i = 0; i < 27; ++ i) travellers[i].clear();

    for (i = 0; i < nm; ++i) {
        const float3 lo = minext_hst[i];
        const float3 hi = maxext_hst[i];

        d[X] = -1 + (lo.x > -XS/2) + (hi.x > XS/2);
        d[Y] = -1 + (lo.y > -YS/2) + (hi.y > YS/2);
        d[Z] = -1 + (lo.z > -ZS/2) + (hi.z > ZS/2);

        type = fabs(d[X]) + fabs(d[Y]) + fabs(d[Z]);

        if (type >= BULK) travellers[0].push_back(i);
        if (type >= FACE) travel::faces(i, d, /**/ travellers);
        if (type >= EDGE) travel::edges(i, d, /**/ travellers);
        if (type >= CORN) travel::cornr(i, d, /**/ travellers);
    }

    for (i = 0; i < 27; ++ i) counts[i] = travellers[i].size();
    return counts[0];
}

void pack(const Particle *pp, const int nv, const std::vector<int> travellers[27], /**/ Particle *spp[27]) {
    /* copy data */
    for (int i = 0; i < 27; ++i) {
        int s = 0; /* start */
        for (int id : travellers[i])
            CC(cudaMemcpyAsync(spp[i] + nv * (s++), pp + nv * id, nv * sizeof(Particle), D2H));
    }
    
    dSync();
}

void post_recv(MPI_Comm cart, const int ank_ne[26], int btc, int btp, /**/ int counts[27], Particle *pp[27], Reqs *rreqs) {
    for (int i = 0; i < 26; ++i) {
        MC(l::m::Irecv(counts + i + 1, 1, MPI_INT, ank_ne[i], btc + i, cart, rreqs->counts + i));
        MC(l::m::Irecv(pp[i + 1], MAX_PART_NUM, datatype::particle, ank_ne[i], btp + i, cart, rreqs->pp + i));
    }
}

void post_send(MPI_Comm cart, const int rnk_ne[26], int btc, int btp, int nv, const int counts[27], const Particle *const pp[27], /**/ Reqs *sreqs) {
    for (int i = 0; i < 26; ++i) {
        const int c = counts[i+1];
        MC(l::m::Isend(counts + i + 1, 1, MPI_INT, rnk_ne[i], btc + i, cart, sreqs->counts + i));
        MC(l::m::Isend(pp[i + 1], c * nv, datatype::particle, rnk_ne[i], btp + i, cart, sreqs->pp + i));
    }
}

int unpack(const int counts[27], const Particle *const rpp[27], const int nv, /**/ Particle *pp) {
    int s = 0; /* start */
    for (int i = 0; i < 27; ++i) {
        const int c = counts[i];
        if (c) CC(cudaMemcpyAsync(pp + s * nv, rpp[i], c * nv * sizeof(Particle), H2D));
        if (i && c) {
            const int d[3] = i2del(i);
            const float3 shift = make_float3(-d[X] * XS, -d[Y] * YS, -d[Z] * ZS);
            KL(dev::shift,
               (k_cnf(c * nv)),
               (shift, c * nv, /**/ pp + s * nv));
        }
        s += c;
    }
    return s;
}

} // sub
} // mcomm
