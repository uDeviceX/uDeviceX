#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "utils/error.h"
#include "utils/imp.h"
#include "parser/imp.h"

#include "d/api.h"
#include "utils/msg.h"
#include "inc/dev.h"

#include "utils/cc.h"
#include "coords/ini.h"

#include "rbc/type.h"
#include "rbc/imp.h"
#include "rbc/force/area_volume/imp.h"
#include "rbc/force/imp.h"

#include "io/mesh/imp.h"
#include "io/off/imp.h"
#include "mpi/wrapper.h"

#include "mpi/glb.h"

static void area_volume_hst(AreaVolume *area_volume, int nc, const Particle *pp, /**/ float *hst) {
    float *dev;
    UC(area_volume_compute(area_volume, nc, pp, /**/ &dev));
    cD2H(hst, dev, 2*nc);
}

static void run0(RbcQuants q) {
    float area, volume, av[2];
    UC(area_volume_hst(q.area_volume, q.nc, q.pp, /**/ av));
    area = av[0]; volume = av[1];
    printf("%g %g\n", area, volume);
}

static void run1(const Coords *coords, OffRead *off, const char *ic, RbcQuants q) {
    rbc_gen_quants(coords, m::cart, off, ic, /**/ &q);
    UC(run0(q));
}

void run(const Coords *coords, const char *cell, const char *ic) {
    RbcQuants q;
    OffRead *off;
    UC(off_read(cell, /**/ &off));
    UC(rbc_ini(off, &q));
    UC(run1(coords, off, ic, q));
    UC(off_fin(off));
    UC(rbc_fin(&q));
}

int main(int argc, char **argv) {
    Coords *coords;
    Config *cfg;

    m::ini(&argc, &argv);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(coords_ini_conf(m::cart, cfg, &coords));
        
    run(coords, "rbc.off", "rbcs-ic.txt");

    UC(coords_fin(coords));
    UC(conf_fin(cfg));
    m::fin();
}
