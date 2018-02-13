#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "algo/force_stat/imp.h"
#include "scheme/time_step/imp.h"

#include "d/api.h"
#include "inc/dev.h"
#include "inc/type.h"
#include "io/txt/imp.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "parser/imp.h"
#include "parser/imp.h"
#include "utils/cc.h"
#include "utils/mc.h"
#include "utils/error.h"
#include "utils/imp.h"
#include "utils/msg.h"

static const char *prog_name = "./udx";
void usg0() {
    fprintf(stderr, "%s i=\\\"force.txt\\\"\n", prog_name);
    fprintf(stderr, "force.txt: f0x f0y f0z\n");
    fprintf(stderr, "           ...\n");
    fprintf(stderr, "           fnx fny fnz\n");
    fprintf(stderr, "test time steps\n");
    exit(0);
}

void usg(int c, char **v) {
    if (c > 1 && same_str(v[1], "-h"))
        usg0();
}

void main0(Config *cfg, int n, const Force *hst) {
    float mass, dt;
    Force *dev;
    TimeStepAccel *accel;
    TimeStep *time_step;
    Dalloc(&dev, n);
    cH2D(dev, hst, n);

    mass = 1.0;
    time_step_ini(cfg, &time_step);
    time_step_accel_ini(&accel);
    time_step_accel_push(accel,     mass, n, dev);
    time_step_accel_push(accel, 0.5*mass, n, dev);

    dt = time_step_dt(time_step, m::cart, accel);
    time_step_log(time_step);
    printf("%g\n", dt);

    time_step_fin(time_step);
    time_step_accel_fin(accel);
    Dfree(dev);
}

int main(int argc, char **argv) {
    const char *i; /* input file */
    const Force *ff;
    int n, rank;
    Config *cfg;
    TxtRead *txt;
    usg(argc, argv);
    m::ini(&argc, &argv);
    MC(m::Comm_rank(m::cart, &rank));
    msg_ini(rank);
    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, /**/ cfg));

    UC(conf_lookup_string(cfg, "i", &i));
    UC(txt_read_ff(i, &txt));

    ff = txt_read_get_ff(txt);
    n  = txt_read_get_n(txt);
    UC(main0(cfg, n, ff));

    UC(txt_read_fin(txt));
    UC(conf_fin(cfg));
    m::fin();
}
