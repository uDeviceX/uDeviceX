#include <mpi.h>
#include <stdio.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/msg.h"
#include "mpi/wrapper.h" /* mini-MPI and -device */
#include "mpi/glb.h"

#include "d/api.h"

#include "utils/error.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "parser/imp.h"
#include "inc/type.h"
#include "inc/dev.h"
#include "dbg/imp.h"
#include "glob/type.h"
#include "glob/ini.h"

const int n = 100;
Particle *pp;
Force *ff;

void alloc() {
    CC(d::Malloc((void**) &pp, n * sizeof(Particle)));
    CC(d::Malloc((void**) &ff, n * sizeof(Force)));
}

void free() {
    CC(d::Free(pp));
    CC(d::Free(ff));
}

namespace dev {

__global__ void fill_bugs(Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Particle p;
    p.r[0] = p.r[1] = p.r[2] = 0;
    p.v[0] = p.v[1] = p.v[2] = 0;

    if (i >= n) return;
    if (i < 1) p.r[0] = 1.5 * XS;
    if (i < 1) p.v[0] = 0.f / 0.f; // nan
    pp[i] = p;
}

__global__ void fill_bugs(Force *ff, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    Force f;
    f.f[0] = f.f[1] = f.f[2] = 0;

    if (i >= n) return;
    if (i < 1) f.f[0] = 1.f / 0.f; // inf
    ff[i] = f;
}
} // dev

void fill_bugs() {
    KL(dev::fill_bugs, (k_cnf(n)), (pp, n));
    KL(dev::fill_bugs, (k_cnf(n)), (ff, n));
}

void check(Coords c, Dbg *dbg) {
    UC(dbg_check_pos    (c, dbg, n, pp));
    UC(dbg_check_vel    (c, dbg, n, pp));
    UC(dbg_check_forces (c, dbg, n, ff));
}

int main(int argc, char **argv) {
    Dbg *dbg;
    Config *cfg;
    Coords coords;
    m::ini(&argc, &argv);
    UC(coords_ini(m::cart, &coords));
    
    UC(conf_ini(&cfg));
    UC(dbg_ini(&dbg));
    UC(conf_read(argc, argv, cfg));
    UC(dbg_set_conf(cfg, dbg));
    
    alloc();
    fill_bugs();
    check(coords, dbg);
    free();
    UC(dbg_fin(dbg));
    UC(conf_fin(cfg));
    UC(coords_fin(&coords));
    m::fin();
}
