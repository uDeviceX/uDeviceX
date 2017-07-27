#include <mpi.h>
#include "m.h"
#include "l/m.h"

#include "common.h"
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

}

void post() {

}

void wait() {

}

void unpack() {

}

} // sub
} // mdstr
